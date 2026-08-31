using JosephsonCircuits
using Plots
gr()

# Same circuit as josephsoncircuits_comparison.jl, but:
#  1. The pump is driven from the P1 end of the line (co-propagating with the
#     signal, which also enters at P1) instead of the P2 end. This is NOT
#     controlled by the sign of v_ratio -- v_pump = abs(v_signal / v_ratio)
#     discards the sign entirely, so direction is purely a circuit-topology
#     choice: which port (3 or 4) is driven with Idc/Ip. Port 3 is built at
#     the P1 end here; port 4 (with the Lg parasitic) is the terminated end
#     at P2.
#  2. Nmodulationharmonics/Npumpharmonics are (3,)/(3,) instead of (1,)/(1,)
#     -- a first-order truncation gave a badly ragged S21 at this gain level.

Z0 = 50.0                          # Ohm, port impedance
nr_cells = 320
cell_size = 10e-6                  # m

omega_cutoff_cfg = 2 * 50 / 530e-3 # rad/ns  -> f_cutoff ~ 30 GHz
omega_j_cfg = 60 * 2 * pi          # rad/ns  -> f_j = 60 GHz
omega_pump_cfg = 13.21 * 2 * pi    # rad/ns  -> f_pump = 13.31 GHz

v_ratio = 1.0                      # |v_pump/v_signal|

Phi_dc_frac = 1/3
Phi_rf_frac = 0.02                 # matches julia_comparison.py's phi_rf_frac

freq_min, freq_max, n_freqs = 1.0, 12.0, 500   # GHz -- matches julia_comparison.py

f_cutoff_GHz = omega_cutoff_cfg / (2 * pi)
f_j_GHz      = omega_j_cfg / (2 * pi)
f_pump_GHz   = omega_pump_cfg / (2 * pi)
println("f_cutoff = $(round(f_cutoff_GHz, digits=3)) GHz, f_j = $(round(f_j_GHz, digits=3)) GHz, ",
        "f_pump = $(round(f_pump_GHz, digits=3)) GHz (co-propagating pump)")
println("Phi_dc = $(Phi_dc_frac) * Phi0, Phi_rf = $(Phi_rf_frac) * Phi0")

Phi0 = 2.0678338484619295e-15      # Wb

    L_cell = 2 * Z0 / omega_cutoff_cfg * 1e-9      # H, target effective series inductance per cell
    C_cell = 2 / (omega_cutoff_cfg * Z0) * 1e-9    # F, shunt-to-ground cap per cell

    # --- FIX 1: bare junction inductance so the BIASED SQUID inductance equals L_cell ---
    # Biased SQUID:  L_squid(Phi_dc) = Lj / cos(phi_dc),  phi_dc = pi*Phi_dc/Phi0 = pi/3.
    # We want L_squid = L_cell, so  Lj = L_cell * cos(pi/3).   (NO extra factor of 2.)
    phi_dc = pi * Phi_dc_frac                        # reduced DC flux = pi/3
    Lj = L_cell * cos(phi_dc) * 2

    Cj  = 1 / ((omega_j_cfg * 1e9)^2 * Lj) / 2
    Ic0 = Phi0 / (2 * pi * Lj)                        # A, per-junction critical current
    println("Lj = $(round(Lj*1e12, digits=2)) pH, Cj = $(round(Cj*1e15, digits=2)) fF, ",
            "Ic0 = $(round(Ic0*1e6, digits=3)) uA, C_cell = $(round(C_cell*1e15, digits=2)) fF")

    v_signal = cell_size * omega_cutoff_cfg * 1e9 / 2   # m/s
    v_pump   = abs(v_signal / v_ratio)                  # m/s
    Lpump = Z0 * cell_size / v_pump                     # H per cell
    Cpump = cell_size / (Z0 * v_pump)                   # F per cell
    println("v_signal = $(round(v_signal, sigdigits=4)) m/s, v_pump = $(round(v_pump, sigdigits=4)) m/s")
    println("Lpump = $(round(Lpump*1e12, digits=2)) pH, Cpump = $(round(Cpump*1e15, digits=2)) fF")

    # --- FIX 2 & 3: make the SQUID-loop inductor and the mutual coupling self-consistent ---
    # The pump couples into the SQUID loop through the mutual inductance
    #   M_actual = kappa * sqrt(Lsmall * Lpump).
    # We want a target coupling (M/Lj = coupling) that sets Lsmall, and then M must be
    # computed from the SAME Lsmall/Lpump/kappa that the circuit's K statement uses,
    # so the flux delivered to the loop is consistent with Idc, Ip below.

    coupling = 0.02
    kappa = 0.999

    # choose Lsmall so that the target mutual inductance M_target = coupling*Lj is realised:
    #   M_target = kappa*sqrt(Lsmall*Lpump)  =>  Lsmall = (M_target/kappa)^2 / Lpump
    M_target = coupling * Lj
    Lsmall = (M_target *kappa)^2 / Lpump            # FIX 3: divisor is Lpump BECAUSE M couples Lsmall<->Lpump
    # now the ACTUAL mutual inductance the circuit sees (used for flux calibration):
    M = kappa * sqrt(Lsmall * Lpump)                 # FIX 2: M consistent with the K statement
    # (by construction M == M_target, but computing it this way guarantees consistency)
    println("M: ", M_target, " - ", M)

    Lg = 20.0e-9

    const magnetic_flux_quantum = 2.0678338484619295e-15
    const reduced_magnetic_flux_quantum = magnetic_flux_quantum / (2*pi)
    optimal_dc_flux = magnetic_flux_quantum * Phi_dc_frac

    Idc = optimal_dc_flux / M_target
    Ip  = Phi_rf_frac * Phi0 / M_target
    println("M = $(round(M*1e12, digits=3)) pH, Lsmall = $(round(Lsmall*1e12, digits=4)) pH")
    println("Idc = $(round(Idc*1e6, digits=2)) uA, Ip = $(round(Ip*1e6, digits=3)) uA")
    @variables Rport C Cj_sym Lj_sym Lpump_sym Cpump_sym kappa_sym Lg_sym Lsmall_sym

    circuit = Tuple{String,String,String,Num}[]
    entry = (elem, n1, n2, value) -> push!(circuit, ("$(elem)$(n1)_$(n2)", "$n1", "$n2", value))

    function build_circuit()
        node = 1

        node_p1 = node
        entry("P", node_p1, 0, 1)
        entry("R", node_p1, 0, Rport)

        node_p3 = node + 1   # pump end near P1 -- driven with DC + AC pump current (co-propagating)
        entry("C", node_p3, 0, Cpump_sym/2)
        entry("P", node_p3, 0, 3)
        entry("R", node_p3, 0, Rport)

        for cell_index in 1:nr_cells
            if cell_index == 1
                entry("C", node, 0, C/2)
            else
                entry("C", node, 0, C)
                entry("C", node+1, 0, Cpump_sym)
            end
            entry("Lj_a", node, node+3, Lj_sym)
            entry("Cj_a", node, node+3, Cj_sym)
            entry("L", node, node+2, Lsmall_sym)
            entry("Lj_b", node+2, node+3, Lj_sym)
            entry("Cj_b", node+2, node+3, Cj_sym)

            entry("L", node+1, node+4, Lpump_sym)
            push!(circuit, ("K$(node)", "L$(node)_$(node+2)", "L$(node+1)_$(node+4)", kappa_sym))

            node += 3
        end

        entry("C", node, 0, C/2)
        entry("P", node, 0, 2)
        entry("R", node, 0, Rport)

        node_p4 = node + 1   # pump end near P2 -- terminated only
        entry("P", node_p4, 0, 4)
        entry("R", node_p4, 0, Rport)
        entry("L", node_p4, 0, Lg_sym)
    end

    build_circuit()

    circuitdefs = Dict(
        Rport => Z0,
        C => C_cell,
        Lj_sym => Lj,
        Cj_sym => Cj,
        Lpump_sym => Lpump,
        Cpump_sym => Cpump,
        Lsmall_sym => Lsmall,
        Lg_sym => Lg,
        kappa_sym => kappa,
    )

# ======

ws = 2 * pi * (range(freq_min, freq_max, length=n_freqs) .+ 0.0137) * 1e9   # tiny offset avoids exact harmonic-coincidence singularities
wp = (2 * pi * f_pump_GHz * 1e9,)

sources = [(mode=(0,), port=3, current=Idc), (mode=(1,), port=3, current=Ip)]

Nmodulationharmonics = (3,)
Npumpharmonics = (3,)

println("Running hbsolve on $(nr_cells) cells / $(2*nr_cells) junctions...")
julia_seconds = @elapsed sol = hbsolve(ws, wp, sources, Nmodulationharmonics,
    Npumpharmonics, circuit, circuitdefs;
    dc=true, threewavemixing=true, fourwavemixing=true,
    switchofflinesearchtol=0.0, alphamin=1e-7, iterations=200)
println("JosephsonCircuits.jl hbsolve: $(round(julia_seconds, digits=3))s")

Phi_delivered = kappa * sqrt(Lsmall * Lpump) * Ip


# outputmode=(-1,) is the amplifier idler: omega_I = omega_s - omega_p (the
# phase-conjugate branch), evaluated at port 2 (the far end, where the
# amplified signal exits).
S21_jl = vec(sol.linearized.S(outputmode=(0,), outputport=2, inputmode=(0,), inputport=1, freqindex=:))   # direct signal transmission (gain)
Sidler_jl = vec(sol.linearized.S(outputmode=(-1,), outputport=2, inputmode=(0,), inputport=1, freqindex=:))  # idler generation (gain)
S11_jl = vec(sol.linearized.S(outputmode=(0,), outputport=1, inputmode=(0,), inputport=1, freqindex=:))
freq_jl = sol.linearized.w / (2 * pi * 1e9)   # GHz


dB(x) = 20 .* log10.(abs.(x) .+ 1e-30)

S21dB = dB(S21_jl)
SidlerdB = dB(Sidler_jl)
ipk21 = argmax(S21dB)
ipkidler = argmax(SidlerdB)
println("S21 gain:     $(round(S21dB[ipk21], digits=2)) dB at $(round(freq_jl[ipk21], digits=3)) GHz")
println("S_idler peak: $(round(SidlerdB[ipkidler], digits=2)) dB at $(round(freq_jl[ipkidler], digits=3)) GHz")

FIGDIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGDIR)


csv_path = joinpath(FIGDIR, "julia_sparams_amplification.csv")
open(csv_path, "w") do f
    println(f, "freq_GHz,S21_re,S21_im,Sidler_re,Sidler_im,S11_re,S11_im")
    for i in eachindex(freq_jl)
        println(f, "$(freq_jl[i]),$(real(S21_jl[i])),$(imag(S21_jl[i])),",
                    "$(real(Sidler_jl[i])),$(imag(Sidler_jl[i])),",
                    "$(real(S11_jl[i])),$(imag(S11_jl[i]))")
    end
end
println("Wrote $(csv_path)")

plt = plot(freq_jl, dB(S21_jl), label="S21 (direct, amplifier gain)", color=:blue,
    xlabel="Signal frequency (GHz)", ylabel="dB",
    title="JosephsonCircuits.jl -- TWPA S-parameters (co-propagating pump)",
    legend=:topright, size=(900, 550),
    xticks=freq_min:1.0:freq_max, yticks=-60:5:30,
    minorgrid=true, minorgridalpha=0.15, gridalpha=0.35)
plot!(plt, freq_jl, dB(Sidler_jl), label="S_idler (phase-conjugate)", color=:red)
plot!(plt, freq_jl, dB(S11_jl), label="S11 (return loss)", color=:green)
hline!(plt, [0], label="0 dB (unity)", color=:black, ls=:dash, lw=1)

outpath_png = joinpath(FIGDIR, "josephsoncircuits_amplification.png")
savefig(plt, outpath_png)
println("\nSaved plot to $(outpath_png).")

try
    outpath_html = joinpath(FIGDIR, "josephsoncircuits_amplification.html")
    savefig(plt, outpath_html)
    println("Also saved an interactive plot to $(outpath_html).")
catch e
    println("(Skipped .html snapshot -- run `Pkg.add([\"PlotlyBase\", \"PlotlyKaleido\"])` ",
            "and switch to `plotly()` if you want one too. The .png above is fully usable without it.)")
end

println()
println("=== Speed ===")
println("  Julia (JosephsonCircuits.jl hbsolve): $(round(julia_seconds, digits=4))s ",
         "($(nr_cells) cells / $(2*nr_cells) junctions)")
