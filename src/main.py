from models import Immittance

def main():
    Zs = Immittance(L=35.5, C=20.1, in_series=False)
    print(Zs.C)

if __name__ == '__main__':
    main()