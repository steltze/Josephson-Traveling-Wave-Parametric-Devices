using JosephsonCircuits
using Plots
gr()

dB(x) = 20 .* log10.(abs.(x) .+ 1e-30)

function build_circuit(nr_cells, Rport, C, Cj_sym, Lj_sym, Lpump_sym, Cpump_sym, kappa_sym, Lg_sym, Lsmall_sym)
    circuit = Tuple{String,String,String,Num}[]
    entry = (elem, n1, n2, value) -> push!(circuit, ("$(elem)$(n1)_$(n2)", "$n1", "$n2", value))
    node = 1

    node_p1 = node
    entry("P", node_p1, 0, 1)
    entry("R", node_p1, 0, Rport)

    node_p4 = node + 1   # pump end near P1 -- terminated only
    entry("P", node_p4, 0, 4)
    entry("R", node_p4, 0, Rport)
    entry("L", node_p4, 0, Lg_sym)

    for cell_index in 1:nr_cells
        if cell_index == 1
            entry("C", node, 0, C/2)
            entry("C", node+1, 0, Cpump_sym/2)
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

    node_p3 = node + 1   # pump end near P2 -- driven with DC + AC pump current
    entry("C", node_p3, 0, Cpump_sym/2)
    entry("P", node_p3, 0, 3)
    entry("R", node_p3, 0, Rport)

    return circuit
end

function main()
    Z0 = 50.0                          # Ohm, port impedance

    nr_cells = parse(Int, get(ENV, "TWPD_NR_CELLS", "320"))
    cell_size = 10e-6                  # m

    omega_cutoff_cfg = 2 * 50 / 530e-3 # rad/ns  -> f_cutoff ~ 30 GHz
    omega_j_cfg = 60 * 2 * pi          # rad/ns  -> f_j = 60 GHz
    omega_pump_cfg = 6.8 * 2 * pi      # rad/ns  -> f_pump = 6.8 GHz

    v_ratio = -2.5                     # negative => pump counter-propagates relative to signal

    Phi_dc_frac = 1 / 3                # DC flux bias, in units of Phi0
    Phi_ac_frac = 0.01                 # AC (pump-induced) flux excursion, in units of Phi0

    freq_min, freq_max = 1.0, 12.0     # GHz, signal sweep
    n_freqs = parse(Int, get(ENV, "TWPD_N_FREQS", "500"))

    Phi0 = 2.0678338484619295e-15      # Wb, magnetic flux quantum

    f_cutoff_GHz = omega_cutoff_cfg / (2 * pi)
    f_j_GHz      = omega_j_cfg / (2 * pi)
    f_pump_GHz   = omega_pump_cfg / (2 * pi)
    println("f_cutoff = $(round(f_cutoff_GHz, digits=3)) GHz, f_j = $(round(f_j_GHz, digits=3)) GHz, ",
            "f_pump = $(round(f_pump_GHz, digits=3)) GHz (counter-propagating: $(v_ratio < 0))")
    println("Phi_dc = $(Phi_dc_frac) * Phi0, Phi_ac = $(Phi_ac_frac) * Phi0")

    # --- Signal (JTWPA) line: unit-cell inductance/capacitance and junction sizing ---

    L_cell = 2 * Z0 / omega_cutoff_cfg * 1e-9      # H, target effective series inductance per cell
    C_cell = 2 / (omega_cutoff_cfg * Z0) * 1e-9    # F, shunt-to-ground capacitance per cell

    phi_dc = pi * Phi_dc_frac
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

    coupling = 0.02
    kappa = 0.999

    M_target = coupling * Lj
    Lsmall = (M_target * kappa)^2 / Lpump
    M = kappa * sqrt(Lsmall * Lpump)
    println("M: ", M_target, " - ", M)

    Lg = 20.0e-9

    optimal_dc_flux = Phi0 * Phi_dc_frac

    Idc = optimal_dc_flux / M_target
    Ip  = Phi_ac_frac * Phi0 / M_target
    println("M = $(round(M*1e12, digits=3)) pH, Lsmall = $(round(Lsmall*1e12, digits=4)) pH")
    println("Idc = $(round(Idc*1e6, digits=2)) uA, Ip = $(round(Ip*1e6, digits=3)) uA")

    @variables Rport C Cj_sym Lj_sym Lpump_sym Cpump_sym kappa_sym Lg_sym Lsmall_sym

    circuit = build_circuit(nr_cells, Rport, C, Cj_sym, Lj_sym, Lpump_sym, Cpump_sym,
                             kappa_sym, Lg_sym, Lsmall_sym)

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

    ws = 2 * pi * range(freq_min, freq_max, length=n_freqs) * 1e9
    wp = (2 * pi * f_pump_GHz * 1e9,)

    sources = [(mode=(0,), port=3, current=Idc), (mode=(1,), port=3, current=Ip)]

    Nmodulationharmonics = (parse(Int, get(ENV, "TWPD_NMOD_HARMONICS", "1")),)
    Npumpharmonics = (parse(Int, get(ENV, "TWPD_NPUMP_HARMONICS", "1")),)


    NREPEATS = parse(Int, get(ENV, "TWPD_BENCH_REPEATS", "2"))
    println("Running hbsolve on $(nr_cells) cells / $(2*nr_cells) junctions ",
            "($(NREPEATS) repeats)...")
    julia_times = Float64[]
    local sol
    for rep in 1:NREPEATS
        t = @elapsed sol = hbsolve(ws, wp, sources, Nmodulationharmonics,
            Npumpharmonics, circuit, circuitdefs;
            dc=true, threewavemixing=true, fourwavemixing=true,
            switchofflinesearchtol=0.0, alphamin=1e-7, iterations=500)
        push!(julia_times, t)
        tag = rep == 1 ? "cold" : "warm"
        println("  run $(rep)/$(NREPEATS) ($(tag)): $(round(t, digits=3))s")
    end

    S31_jl = vec(sol.linearized.S(outputmode=(1,), outputport=1, inputmode=(0,), inputport=1, freqindex=:))
    S21_jl = vec(sol.linearized.S(outputmode=(0,), outputport=2, inputmode=(0,), inputport=1, freqindex=:))
    S11_jl = vec(sol.linearized.S(outputmode=(0,), outputport=1, inputmode=(0,), inputport=1, freqindex=:))
    freq_jl = sol.linearized.w / (2 * pi * 1e9)   # GHz

    S21dB = dB(S21_jl)
    S31dB = dB(S31_jl)
    imin = argmin(S21dB)
    ipk = argmax(S31dB)
    println("S21 notch: $(round(S21dB[imin], digits=2)) dB at $(round(freq_jl[imin], digits=3)) GHz")
    println("S31 peak:  $(round(S31dB[ipk], digits=2)) dB at $(round(freq_jl[ipk], digits=3)) GHz")

    FIGDIR = joinpath(@__DIR__, "..", "figures")
    mkpath(FIGDIR)

    csv_path = joinpath(FIGDIR, "julia_sparams.csv")
    open(csv_path, "w") do f
        println(f, "freq_GHz,S21_re,S21_im,S31_re,S31_im,S11_re,S11_im")
        for i in eachindex(freq_jl)
            println(f, "$(freq_jl[i]),$(real(S21_jl[i])),$(imag(S21_jl[i])),",
                        "$(real(S31_jl[i])),$(imag(S31_jl[i])),",
                        "$(real(S11_jl[i])),$(imag(S11_jl[i]))")
        end
    end
    println("Wrote $(csv_path)")

    timings_path = joinpath(FIGDIR, "julia_timings.csv")
    open(timings_path, "w") do f
        println(f, "run,tag,seconds")
        for (rep, t) in enumerate(julia_times)
            tag = rep == 1 ? "cold" : "warm"
            println(f, "$(rep),$(tag),$(t)")
        end
    end
    println("Wrote $(timings_path)")

    plt = plot(freq_jl, dB(S31_jl), label="S31 (conversion)", color=:blue,
        xlabel="Signal frequency (GHz)", ylabel="dB",
        title="JosephsonCircuits.jl -- TWPC S-parameters",
        legend=:bottomright, size=(900, 550),
        xticks=freq_min:0.5:freq_max, yticks=-100:5:10,
        minorgrid=true, minorgridalpha=0.15, gridalpha=0.35)
    plot!(plt, freq_jl, dB(S21_jl), label="S21 (direct)", color=:red)
    plot!(plt, freq_jl, dB(S11_jl), label="S11 (return loss)", color=:green)

    outpath_png = joinpath(FIGDIR, "josephsoncircuits_comparison.png")
    savefig(plt, outpath_png)
    println("\nSaved plot to $(outpath_png).")

    warm_times = julia_times[2:end]
    warm_str = isempty(warm_times) ? "n/a" : "$(round(sum(warm_times)/length(warm_times), digits=4))s"
    println()
    println("hbsolve cold (run 1): $(round(julia_times[1], digits=4))s, ",
             "warm (mean of runs 2:$(NREPEATS)): $(warm_str) ",
             "($(nr_cells) cells / $(2*nr_cells) junctions)")

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
