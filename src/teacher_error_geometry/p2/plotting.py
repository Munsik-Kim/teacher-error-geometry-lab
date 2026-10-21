"""Two static P2 figures, generated only from saved summaries."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..artifacts import read_json, sha256, write_json


def figures(summary,output):
    summary=Path(summary); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    overview=read_json(summary/"overview.json"); rows=overview["methods"]; cfg=overview["config"]
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "svg.hashsalt":"teacher-error-geometry-p2","figure.dpi":120})
    paths=[]
    def save(fig,name):
        fig.tight_layout()
        for extension in ("png","svg"):
            path=output/f"{name}.{extension}"
            fig.savefig(path,dpi=170,bbox_inches="tight",metadata={"Date":None} if extension=="svg" else {})
            paths.append(path)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    x=np.arange(4)
    ax.plot(x-.06,[r["theory"] for r in rows],"s",ms=8,mfc="none",color="#202020",label="theory expectation")
    ax.errorbar(x+.06,[r["mean"] for r in rows],yerr=[1.96*r["mc_se"] for r in rows],fmt="o",ms=6,capsize=5,
                color="#2166ac",label="MC mean +/- pointwise 95%")
    ax.set_xticks(x,["FO\nfull / outcome","FT\nfull / teacher","RO\nrepresentation / outcome","RM\nrepresentation / mixed"])
    ax.set_ylabel("population squared risk against true signal X beta")
    ax.set_title(f"Gaussian paper anchor: d=5, r=3, n=27, lambda=0.45; R={cfg['repetitions']:,}")
    ax.grid(axis="y",alpha=.2); ax.legend(loc="best")
    for i,r in enumerate(rows):
        ax.annotate(f"MC {r['mean']:.6f}\ntheory {r['theory']:.6f}",(i,max(r["mean"],r["theory"])+1.96*r["mc_se"]),
                    xytext=(0,12),textcoords="offset points",ha="center",fontsize=9)
    low=min(r["mean"]-1.96*r["mc_se"] for r in rows); high=max(r["mean"]+1.96*r["mc_se"] for r in rows)
    ax.set_ylim(low-.035,high+.06)
    save(fig,"01_four_methods_truth_risk")

    terms=["representation_loss","teacher_bias","cross_term","estimation_quadratic"]
    rm={r["component"]:r for r in overview["components"] if r["method"]=="RM"}
    truth=next(r for r in rows if r["method"]=="RM")
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    ax=axes[0]; x=np.arange(4)
    ax.bar(x,[rm[t]["theory"] for t in terms],width=.55,color="#c9d6df",label="expected contribution")
    ax.errorbar(x,[rm[t]["mean"] for t in terms],yerr=[1.96*rm[t]["mc_se"] for t in terms],fmt="o",color="#b54822",
                capsize=4,label="actual component mean +/-95%")
    ax.axhline(0,color="black",lw=.7)
    ax.set_xticks(x,["representation\nloss","projected teacher\nbias","sample cross\n(signed)","estimation\nquadratic"])
    ax.set_ylabel("contribution to truth risk"); ax.set_title("RM: expectation and fitted components")
    for i,t in enumerate(terms):
        ax.annotate(f"{rm[t]['mean']:+.7g}",(i,rm[t]["mean"]),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9)
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=.2)
    ax=axes[1]
    values=[truth["theory"],truth["mean"],sum(rm[t]["mean"] for t in terms)]
    ax.bar(np.arange(3),values,color=["#c9d6df","#2166ac","#b54822"],width=.55)
    ax.errorbar([1],[truth["mean"]],yerr=[1.96*truth["mc_se"]],fmt="none",ecolor="black",capsize=5)
    ax.set_xticks(range(3),["theory\ntotal","MC fitted\ntruth risk","sum of actual\ncomponent means"])
    ax.set_ylabel("RM total population risk"); ax.set_title("The finite MC cross term is retained")
    for i,value in enumerate(values):ax.text(i,value+.005,f"{value:.8f}",ha="center",fontsize=10)
    ax.set_ylim(0,max(values)*1.2); ax.grid(axis="y",alpha=.2)
    fig.suptitle(f"RM n=27, lambda=.45: observed mean cross={rm['cross_term']['mean']:+.8g}; expectation=0",y=1.03)
    save(fig,"02_rm_decomposition")
    result={"figure_count":2,"files":[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha256(p)} for p in paths],
            "source":"saved summary only; no refits or test data generated","intervals":"pointwise normal 95%"}
    write_json(output/"manifest.json",result)
    return result
