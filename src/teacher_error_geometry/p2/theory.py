"""Population projection and expected risks, separate from fitted evaluator."""
from fractions import Fraction as F
import numpy as np
from .model import arrays


def projection(config):
    p=arrays(config); S,A,beta,delta=(p[k] for k in ("Sigma","A","beta","delta")); gamma=beta+delta
    C=A@S@A.T
    wb=np.linalg.solve(C,A@S@beta); wg=np.linalg.solve(C,A@S@gamma)
    ub=beta-A.T@wb; ug=gamma-A.T@wg
    return {"C":C,"w_beta":wb,"w_gamma":wg,"u_beta":ub,"u_gamma":ug,
            "B":float(delta@S@delta),"B_A":float(ub@S@ub),
            "D_A":float((wg-wb)@C@(wg-wb)),"G_A":float(ug@S@ug),"S_A":float(ub@S@ug)}


def expectation(config):
    g=projection(config); n,d,r=config["n"],config["d"],config["r"]
    if n<=d+1 or n<=r+1:
        raise ValueError("Expected risk requires n>d+1 and n>r+1")
    s,t,o,l=(config[k] for k in ("sigma2","tau2","omega","lambda"))
    U=g["B_A"]+s; V=g["G_A"]+t; H=g["S_A"]+o
    bracket=(1-l)**2*U+l*l*V+2*l*(1-l)*H
    variance=r/(n-r-1)*bracket
    risks={"FO":s*d/(n-d-1),"FT":g["B"]+t*d/(n-d-1),
           "RO":g["B_A"]+r*(g["B_A"]+s)/(n-r-1),
           "RM":g["B_A"]+l*l*g["D_A"]+variance}
    components={
      "FO":{"representation_loss":0.,"teacher_bias":0.,"cross_term":0.,"estimation_quadratic":risks["FO"]},
      "FT":{"representation_loss":0.,"teacher_bias":g["B"],"cross_term":0.,"estimation_quadratic":t*d/(n-d-1)},
      "RO":{"representation_loss":g["B_A"],"teacher_bias":0.,"cross_term":0.,"estimation_quadratic":r*U/(n-r-1)},
      "RM":{"representation_loss":g["B_A"],"teacher_bias":l*l*g["D_A"],"cross_term":0.,"estimation_quadratic":variance}}
    return {"risks":risks,"components":components,"rm_bracket":bracket,"residual_coupling":H}


def exact_anchor():
    """Independent scalar/Fraction path, fixed decimal parameters, no matrices."""
    beta=list(map(F,[".9","-.6",".5",".2","-.2"]))
    delta=list(map(F,[".22","-.13",".10",".35","-.20"]))
    B=sum(x*x for x in delta); BA=sum(x*x for x in beta[3:]); DA=sum(x*x for x in delta[:3])
    GA=sum((b+d)**2 for b,d in zip(beta[3:],delta[3:])); SA=sum(b*(b+d) for b,d in zip(beta[3:],delta[3:]))
    s,t,o,l=F("1.3"),F(".7"),F(".22"),F(".45")
    bracket=(1-l)**2*(BA+s)+l*l*(GA+t)+2*l*(1-l)*(SA+o)
    risks={"FO":s*F(5,21),"FT":B+t*F(5,21),"RO":BA+F(3,23)*(BA+s),"RM":BA+l*l*DA+F(3,23)*bracket}
    return {"geometry":{k:float(v) for k,v in zip(("B","B_A","D_A","G_A","S_A"),(B,BA,DA,GA,SA))},
            "fractions":{k:str(v) for k,v in risks.items()},"risks":{k:float(v) for k,v in risks.items()},
            "rm_bracket":float(bracket),"rm_variance":float(F(3,23)*bracket)}
