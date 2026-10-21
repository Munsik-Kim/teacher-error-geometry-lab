"""Gaussian design, correlated noise and shared targets. No data adjustment."""
import hashlib
import numpy as np

METHODS = ("FO", "FT", "RO", "RM")


def noise_factor(sigma2, tau2, omega):
    if not np.isfinite([sigma2,tau2,omega]).all() or sigma2 < 0 or tau2 < 0:
        raise ValueError("Noise variances must be finite nonnegative; omega is covariance")
    if sigma2 == 0:
        if omega != 0:
            raise ValueError("Zero outcome variance requires zero covariance")
        return np.array([[0.,0.],[0.,np.sqrt(tau2)]], dtype=np.float64)
    residual = tau2 - omega*omega/sigma2
    if residual < 0:
        raise ValueError("Invalid noise covariance: negative conditional variance")
    return np.array([[np.sqrt(sigma2),0.],[omega/np.sqrt(sigma2),np.sqrt(residual)]], dtype=np.float64)


def arrays(config):
    return {k:np.asarray(config[k],dtype=np.float64) for k in ("Sigma","A","beta","delta")}


def validate(config):
    d,r,n = config["d"],config["r"],config["n"]
    if any(int(x)!=x or x<1 for x in (d,r,n)) or r>d or n<=d+3:
        raise ValueError("Require integer 1<=r<=d, n>d+3 for finite Gaussian risk variance")
    a=arrays(config)
    if a["Sigma"].shape!=(d,d) or a["A"].shape!=(r,d) or a["beta"].shape!=(d,) or a["delta"].shape!=(d,):
        raise ValueError("Invalid parameter shapes")
    if not all(np.isfinite(x).all() for x in a.values()) or not np.array_equal(a["Sigma"],a["Sigma"].T):
        raise ValueError("Nonfinite parameters or nonsymmetric covariance")
    np.linalg.cholesky(a["Sigma"])
    if np.linalg.matrix_rank(a["A"]) != r:
        raise ValueError("Representation must have full row rank")
    noise_factor(config["sigma2"],config["tau2"],config["omega"])
    if not np.isfinite(config["lambda"]) or not 0<=config["lambda"]<=1:
        raise ValueError("lambda must be fixed in [0,1]")
    if config["dtype"]!="float64" or tuple(config["methods"])!=METHODS:
        raise ValueError("Unsupported precision or method mapping")
    if int(config["repetitions"])!=config["repetitions"] or config["repetitions"]<2:
        raise ValueError("At least two repetitions required")
    if int(config["batch_size"])!=config["batch_size"] or config["batch_size"]<1 or config["wall_limit_seconds"]<=0:
        raise ValueError("Invalid execution budget")
    streams=config["rng"]["streams"]
    if set(streams)!={"X","E","H","test_mc"} or len(set(streams.values()))!=4:
        raise ValueError("X/E/H/test streams must be distinct")
    expected=[[METHODS[i],METHODS[j]] for i in range(4) for j in range(i+1,4)]
    if config["paired_contrasts"]!=expected:
        raise ValueError("Exactly six ordered-sign unordered pairs required")
    return a


def seed(config, stream, replicate):
    values=[config["root_seed"],config["rng"]["streams"][stream],replicate]
    if any(int(v)!=v or v<0 for v in values):
        raise ValueError("Nonnegative integer seed components required")
    return [int(v) for v in values]


def standard(config, stream, replicate, shape):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed(config,stream,replicate)))).standard_normal(shape,dtype=np.float64)


def generate(config, replicate):
    p=arrays(config); n,d=config["n"],config["d"]
    X=standard(config,"X",replicate,(n,d)) @ np.linalg.cholesky(p["Sigma"]).T
    E=standard(config,"E",replicate,n); H=standard(config,"H",replicate,n)
    factor=noise_factor(config["sigma2"],config["tau2"],config["omega"])
    epsilon=factor[0,0]*E
    xi=factor[1,0]*E+factor[1,1]*H
    Y,T,V=targets(X,epsilon,xi,p["beta"],p["delta"],config["lambda"])
    return {"X":X,"Z":X@p["A"].T,"epsilon":epsilon,"xi":xi,"Y":Y,"T":T,"V":V}


def targets(X,epsilon,xi,beta,delta,weight):
    Y=X@beta+epsilon
    T=X@(beta+delta)+xi
    return Y,T,(1-weight)*Y+weight*T


def array_hash(*arrays_to_hash):
    digest=hashlib.sha256()
    for x in arrays_to_hash:
        digest.update(np.asarray(x,dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def hashes(data):
    return {"x_sha256":array_hash(data["X"]),"z_sha256":array_hash(data["Z"]),
            "noise_sha256":array_hash(data["epsilon"],data["xi"]),
            "y_sha256":array_hash(data["Y"]),"t_sha256":array_hash(data["T"]),
            "v_sha256":array_hash(data["V"]),"targets_sha256":array_hash(data["Y"],data["T"],data["V"])}


def assigned(data, method):
    design=data["X"] if method in ("FO","FT") else data["Z"]
    target=data[{"FO":"Y","FT":"T","RO":"Y","RM":"V"}[method]]
    return design,target
