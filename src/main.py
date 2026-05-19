from models import Immittance

def main():
    Zs = Immittance(L=35.5, C=20.1, in_series=False)
    print(Zs.C)



#     # 3 JJs in series
# Series(JJ(L), JJ(L), JJ(L))          # == JJ(3L)

# # 3 JJs in parallel  
# Parallel(Series(JJ(L), JJ(L), JJ(L)), Capacitor(C_g))        # == JJ(L/3)

# # Two-mode Δ shunt
# Parallel(Capacitor(C_g), Capacitor(2*C_i))

# # SNAIL (future — just add a SNAIL class with phi^3 potential)
# Series(SNAIL(L_small, L_large, n=3, flux=0.5))

if __name__ == '__main__':
    main()