import argparse, time, json, random, httpx
def gen_event(i, anomaly_p=0.03):
    rng=random.Random(time.time_ns())
    t=rng.gauss(60,3); h=rng.gauss(35,4); s=rng.gauss(75,5)
    if rng.random()<anomaly_p: t+=rng.choice([+12,-12]); h+=rng.choice([+15,-15]); s+=rng.choice([+20,-20])
    return {"machine_id":f"A-{i%20:02d}","temperature_c":round(t,2),"humidity_pct":round(h,2),"sound_db":round(s,2)}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--num",type=int,default=20); p.add_argument("--rate_hz",type=float,default=1.0)
    p.add_argument("--send",type=str,default=None); p.add_argument("--batch",type=int,default=0)
    a=p.parse_args(); period=1.0/max(1e-6,a.rate_hz); client=httpx.Client(timeout=5.0) if a.send else None; buf=[]
    for i in range(a.num):
        ev=gen_event(i)
        if client:
            if a.batch>0:
                buf.append(ev)
                if len(buf)>=a.batch: r=client.post(a.send,json=buf); print("POST batch ->",r.status_code,r.text[:200]); buf.clear()
            else:
                r=client.post(a.send,json=ev); print("POST single ->",r.status_code,r.text.strip())
        else: print(json.dumps(ev))
        time.sleep(period)
    if client and buf: r=client.post(a.send,json=buf); print("POST final batch ->",r.status_code,r.text[:200])
if __name__=="__main__": main()
