import json
from pathlib import Path

files=["autonomous_result_iris.json","autonomous_result_telco.json","autonomous_result_housing.json"]
out=[]
for f in files:
    p=Path(f)
    if not p.exists():
        out.append({"dataset":f, "error":"missing file"})
        continue
    d=json.loads(p.read_text())
    out.append({
        "dataset": f.replace("autonomous_result_","" ).replace('.json',''),
        "status": "PASS" if len(d.get('stage_errors',[]))==0 else "FAIL",
        "runtime": round(d.get('total_runtime',0),1),
        "best_model": d.get('model_results',{}).get('best_model'),
        "prediction_possible": bool(d.get('production_model')),
        "pdf_report": bool(d.get('final_report',{}).get('path')),
        "errors": len(d.get('stage_errors',[]))
    })
print(json.dumps(out))
