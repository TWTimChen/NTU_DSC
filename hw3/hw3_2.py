import ParticleSworm as PS

def main():
    def f1(x):
        assert len(x)==1
        return(-x**5 + 5*x**3 + 20*x - 5)

    x = [[-2], [0], [1] ,[3]]
    
    ps = PS.ParticleSworm(xLow=[-4], xUp=[4])
    ps.evaluation = f1
    ps.fit(x)
    ps.show()

main()