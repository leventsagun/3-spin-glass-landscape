-- 3-spin spherical spin glass model

-- environment is chosen as one of the below ensembles
function FixEnv(N, distribution)
    if distribution == 1 then
        surface = torch.CudaTensor(N, N, N):normal()
    elseif distribution == 2 then
        surface = (torch.CudaTensor(N, N, N):bernoulli() - 0.5)*2
    elseif distribution == 3 then
        surface = torch.CudaTensor(N, N, N):uniform(0,math.sqrt(12)) - math.sqrt(12)/2
    elseif distribution == 4 then
        surface = torch.CudaTensor(N, N, N):exponential(1)-1
    end

    return surface
end

-- value of the hamiltonian given couplings and a spin configuration
function EnergyTensor(spins, couplings)

    N=couplings:size()[1]

    w2 = torch.CudaTensor(N)
    H = torch.CudaTensor(N, N, N)
    ger_tmp = torch.CudaTensor(N, N)
    ger_tmp2 = torch.CudaTensor(N^2, N)

    w2:copy(spins):resize(N, 1)
    ger_tmp:addmm(0, 1, w2, spins:resize(1, N))
    ger_tmp2:addmm(0, 1, ger_tmp:resize(N^2, 1), spins:resize(1, N))
    spins:resize(N)
    H:copy(couplings)
    H:cmul(ger_tmp2)

    return H
end

-- given surface, pick a random initial point, and run gradient descent
function SpinItGD(Surface)

    N=Surface:size()[1]

    w = torch.CudaTensor(N):normal()
    gw = torch.CudaTensor(N)
    w:mul(math.sqrt(N)/(w:norm()))
    wInitial=w

    H=EnergyTensor(w, Surface)
    EnergyOld=H:sum()/N
    EnergyNew=EnergyOld-10
    gw:fill(1)

    time=0
    Times={}
    Energies={}
    c=1

    while gw:norm() > precision[#precision] do
        time=time+1
        gamma = learningRate / (1 + time * learningRateDecay)

        gw:zero()
        gw:add(H:sum(1):sum(3)):add(H:sum(2):sum(1)):add(H:sum(3):sum(2))
        gw:cdiv(w):div(N) --invlves division by a possibly tiny number!!!
        gw:add(-gw:dot(w)/N, w)
        w:add(-gamma, gw)
        w:div(w:norm()/math.sqrt(N))
        H=EnergyTensor(w, Surface)
        EnergyOld=EnergyNew
        EnergyNew=H:sum()/N
        print(time)
        print(gw:norm())
        print(EnergyNew)

        --collects values at certain precision levels along the way
        if gw:norm() < precision[c] then
          Times[c]=time
          Energies[c]=EnergyNew
          c=c+1
        end

        collectgarbage()
    end
end
