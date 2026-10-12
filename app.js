(function(){
"use strict";

/* ---------- Random numbers (seeded, repeatable) ---------- */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function makeGauss(rng){let spare=null;return function(){if(spare!==null){const s=spare;spare=null;return s;}let u,v,s;do{u=rng()*2-1;v=rng()*2-1;s=u*u+v*v;}while(s>=1||s===0);const m=Math.sqrt(-2*Math.log(s)/s);spare=v*m;return u*m;};}

/* ---------- Initialization schemes ---------- */
const INITS={
  zeros:  {label:"All zeros",              fn:()=>0,                                     variance:"0",                    sd:()=>0,
           built:"Nothing", fails:"Every neuron in a layer computes the same thing and gets the same update, so they never become different. Gradients die after the last layer."},
  small:  {label:"Tiny normal (σ = 0.01)", fn:(fi,fo,g)=>0.01*g(),                      variance:"0.0001 (fixed)",       sd:()=>0.01,
           built:"Shallow networks", fails:"Signal shrinks by a constant factor per layer; deep networks receive almost no gradient."},
  large:  {label:"Standard normal (σ = 1)", fn:(fi,fo,g)=>g(),                          variance:"1 (fixed)",            sd:()=>1,
           built:"Nothing in particular", fails:"Sums of many inputs blow up; tanh and sigmoid saturate, ReLU networks explode."},
  lecun:  {label:"LeCun normal",           fn:(fi,fo,g)=>g()*Math.sqrt(1/fi),           variance:"1 / n<sub>in</sub>",   sd:(n)=>Math.sqrt(1/n),
           built:"tanh, SELU", fails:"Preserves the forward pass only; gradients can still drift in non-square layers."},
  xavierU:{label:"Xavier uniform",         fn:(fi,fo,g,r)=>(r()*2-1)*Math.sqrt(6/(fi+fo)), variance:"2 / (n<sub>in</sub> + n<sub>out</sub>)", sd:(n)=>Math.sqrt(1/n),
           built:"tanh, linear, softsign", fails:"Under ReLU the signal halves in energy each layer and slowly fades."},
  xavierN:{label:"Xavier normal",          fn:(fi,fo,g)=>g()*Math.sqrt(2/(fi+fo)),      variance:"2 / (n<sub>in</sub> + n<sub>out</sub>)", sd:(n)=>Math.sqrt(1/n),
           built:"tanh, linear, softsign", fails:"Same as Xavier uniform; the distribution shape matters far less than its variance."},
  he:     {label:"He / Kaiming normal",    fn:(fi,fo,g)=>g()*Math.sqrt(2/fi),           variance:"2 / n<sub>in</sub>",   sd:(n)=>Math.sqrt(2/n),
           built:"ReLU, Leaky ReLU", fails:"With tanh it runs slightly hot, pushing more units toward saturation."}
};
const ACTS={
  tanh:   {label:"tanh",    f:Math.tanh, d:z=>{const t=Math.tanh(z);return 1-t*t;}, range:[-1,1]},
  relu:   {label:"ReLU",    f:z=>z>0?z:0, d:z=>z>0?1:0, range:[0,3]},
  sigmoid:{label:"sigmoid", f:z=>1/(1+Math.exp(-z)), d:z=>{const s=1/(1+Math.exp(-z));return s*(1-s);}, range:[0,1]},
  linear: {label:"linear (none)", f:z=>z, d:()=>1, range:[-3,3]}
};

/* ---------- Helpers ---------- */
const $=id=>document.getElementById(id);
function cssv(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim();}
function hexToRgb(h){h=h.replace('#','');if(h.length===3)h=h.split('').map(c=>c+c).join('');const n=parseInt(h,16);return[(n>>16)&255,(n>>8)&255,n&255];}
function std(arr){let m=0;const n=arr.length;for(let i=0;i<n;i++)m+=arr[i];m/=n;let v=0;for(let i=0;i<n;i++){const d=arr[i]-m;v+=d*d;}return Math.sqrt(v/n);}
function setupCanvas(cv,cssH){const dpr=window.devicePixelRatio||1;const w=Math.max(10,cv.clientWidth);const h=cssH||cv.clientHeight;cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);if(cssH)cv.style.height=cssH+'px';const ctx=cv.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return{ctx,w,h};}
function fmtSci(x){if(!isFinite(x))return x>0||isNaN(x)?"∞":"0";if(x===0)return"0";const a=Math.abs(x);if(a>=0.01&&a<1000)return x.toPrecision(2).replace(/\.?0+$/,'')||"0";const e=Math.floor(Math.log10(a));const m=x/Math.pow(10,e);return m.toFixed(1)+"e"+e;}
const displayFont="12px 'Segoe UI', system-ui, sans-serif";

/* ---------- Populate selects ---------- */
const l1Init=$('l1-init'),l1Act=$('l1-act');
const l1InitKeys=['zeros','small','large','lecun','xavierU','xavierN','he'];
l1InitKeys.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=INITS[k].label;l1Init.appendChild(o);});
Object.keys(ACTS).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=ACTS[k].label;l1Act.appendChild(o);});
l1Init.value='xavierN';l1Act.value='tanh';

/* ======================================================
   LAB 1: signal propagation at initialization
   ====================================================== */
let l1Seed=7,l1Result=null,l1Timer=null;

function simulate(initKey,actKey,depth,width,seed){
  const B=128,rng=mulberry32(seed),g=makeGauss(rng),init=INITS[initKey].fn,act=ACTS[actKey];
  let a=new Float64Array(B*width);for(let i=0;i<a.length;i++)a[i]=g();
  const Ws=[],Zs=[],As=[];
  for(let l=0;l<depth;l++){
    const W=new Float64Array(width*width);for(let i=0;i<W.length;i++)W[i]=init(width,width,g,rng);
    const z=new Float64Array(B*width);
    for(let b=0;b<B;b++){const ao=b*width;for(let j=0;j<width;j++){let s=0;const wo=j*width;for(let i=0;i<width;i++)s+=a[ao+i]*W[wo+i];z[ao+j]=s;}}
    const out=new Float64Array(B*width);for(let i=0;i<out.length;i++)out[i]=act.f(z[i]);
    Ws.push(W);Zs.push(z);As.push(out);a=out;
  }
  let da=new Float64Array(B*width);for(let i=0;i<da.length;i++)da[i]=g();
  const gradStd=new Array(depth);
  for(let l=depth-1;l>=0;l--){
    gradStd[l]=std(da);
    const z=Zs[l],W=Ws[l],dz=new Float64Array(B*width);
    for(let i=0;i<dz.length;i++)dz[i]=da[i]*act.d(z[i]);
    const nd=new Float64Array(B*width);
    for(let b=0;b<B;b++){const ao=b*width;for(let j=0;j<width;j++){const d=dz[ao+j];if(d===0)continue;const wo=j*width;for(let i=0;i<width;i++)nd[ao+i]+=d*W[wo+i];}}
    da=nd;
  }
  // saturation in final layer
  const last=As[depth-1];let sat=0;
  if(actKey==='tanh'){for(let i=0;i<last.length;i++)if(Math.abs(last[i])>0.97)sat++;}
  else if(actKey==='sigmoid'){for(let i=0;i<last.length;i++)if(last[i]<0.03||last[i]>0.97)sat++;}
  else if(actKey==='relu'){for(let i=0;i<last.length;i++)if(last[i]===0)sat++;}
  return{As,actStd:As.map(std),gradStd,sat:sat/last.length,depth,width,actKey,initKey};
}

function pickLayers(depth){
  const n=Math.min(6,depth),out=[];
  for(let k=0;k<n;k++)out.push(Math.round(k*(depth-1)/(n-1||1)));
  return [...new Set(out)];
}

function drawHist(cv,data,range){
  const {ctx,w,h}=setupCanvas(cv,84);
  const bins=28,[lo,hi]=range,counts=new Array(bins).fill(0);
  let nonFinite=0;
  for(let i=0;i<data.length;i++){let v=data[i];if(!isFinite(v)){nonFinite++;continue;}let k=Math.floor((v-lo)/(hi-lo)*bins);if(k<0)k=0;if(k>=bins)k=bins-1;counts[k]++;}
  const max=Math.max(1,...counts);
  ctx.clearRect(0,0,w,h);
  const bw=w/bins,act=cssv('--act'),warn=cssv('--l2');
  const edgeHeavy=(counts[0]+counts[bins-1])/data.length;
  for(let k=0;k<bins;k++){
    const bh=Math.max(counts[k]>0?1.5:0,(counts[k]/max)*(h-14));
    const isEdge=(k===0||k===bins-1)&&counts[k]/data.length>0.08&&!(range[0]===0&&k===0);
    ctx.fillStyle=isEdge?warn:act;
    ctx.fillRect(k*bw+0.5,h-12-bh,Math.max(1,bw-1),bh);
  }
  ctx.fillStyle=cssv('--muted');ctx.font="10px Consolas, ui-monospace, monospace";
  ctx.textAlign='left';ctx.fillText(String(lo),0,h-1);
  ctx.textAlign='right';ctx.fillText(String(hi),w,h-1);
  ctx.strokeStyle=cssv('--line');ctx.beginPath();ctx.moveTo(0,h-12.5);ctx.lineTo(w,h-12.5);ctx.stroke();
  return edgeHeavy;
}

function drawStdChart(res){
  const cv=$('l1-chart');const H=window.innerWidth<600?200:240;
  const {ctx,w,h}=setupCanvas(cv,H);
  ctx.clearRect(0,0,w,h);
  const padL=46,padR=12,padT=12,padB=42,pw=w-padL-padR,ph=h-padT-padB;
  const floor=1e-16,ceil=1e16;
  const clamp=v=>(!isFinite(v)||isNaN(v))?ceil:Math.min(ceil,Math.max(floor,v));
  const series=[res.actStd.map(clamp),res.gradStd.map(clamp)];
  let lo=Infinity,hi=-Infinity;series.forEach(s=>s.forEach(v=>{const L=Math.log10(v);lo=Math.min(lo,L);hi=Math.max(hi,L);}));
  lo=Math.floor(Math.min(lo,-1));hi=Math.ceil(Math.max(hi,1));
  const span=hi-lo,stepD=span>12?4:span>6?2:1;
  const X=i=>padL+(res.depth===1?pw/2:i*pw/(res.depth-1));
  const Y=v=>padT+ph-(Math.log10(v)-lo)/span*ph;
  ctx.font=displayFont;ctx.fillStyle=cssv('--muted');ctx.strokeStyle=cssv('--line');ctx.lineWidth=1;
  ctx.textAlign='right';ctx.textBaseline='middle';
  for(let e=lo;e<=hi;e+=stepD){const y=Y(Math.pow(10,e));ctx.globalAlpha=e===0?1:0.6;ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(w-padR,y);ctx.stroke();ctx.globalAlpha=1;ctx.fillText(e===0?"1":"1e"+e,padL-6,y);}
  ctx.textAlign='center';ctx.textBaseline='top';
  const every=Math.ceil(res.depth/12);
  for(let i=0;i<res.depth;i+=every)ctx.fillText(String(i+1),X(i),h-padB+8);
  ctx.textAlign='center';ctx.fillText('Layer',padL+pw/2,h-padB+24);
  const cols=[cssv('--act'),cssv('--grad')];
  series.forEach((s,si)=>{
    ctx.strokeStyle=cols[si];ctx.lineWidth=2.5;ctx.beginPath();
    s.forEach((v,i)=>{const x=X(i),y=Y(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    ctx.fillStyle=cols[si];s.forEach((v,i)=>{ctx.beginPath();ctx.arc(X(i),Y(v),3,0,Math.PI*2);ctx.fill();});
  });
}

function insightFor(res){
  const {actStd,gradStd,depth,sat,actKey,initKey}=res;
  const first=actStd[0],last=actStd[depth-1],gIn=gradStd[0],gOut=gradStd[depth-1];
  if(initKey==='zeros'){
    return{cls:'bad',text:"Every weight is zero, so every unit in a layer computes exactly the same value"+(actKey==='sigmoid'?" (0.5 for sigmoid)":"")+". The gradient flowing back into earlier layers is also zero, and even where updates do happen, all units in a layer get identical updates and stay identical forever. Symmetry has to be broken with randomness."};
  }
  const ratio=last/(first||1e-300);
  const gRatio=gIn/(gOut||1e-300);
  const parts=[];let cls='';
  if(!isFinite(last)||ratio>100){parts.push(`The forward signal grows about ${fmtSci(ratio)}× from layer 1 to layer ${depth}.`);cls='bad';}
  else if(ratio<0.01){parts.push(`The forward signal shrinks by a factor of about ${fmtSci(1/ratio)} between layer 1 and layer ${depth}; the last layers barely see the input.`);cls='bad';}
  else{parts.push(`The forward signal holds its scale: its spread changes by only ${ratio.toFixed(2)}× across ${depth} layers.`);}
  if(actKey==='tanh'&&sat>0.3){parts.push(`About ${Math.round(sat*100)}% of the final layer's tanh units sit pinned near ±1, where their slope is close to zero (the red edge bars).`);cls='bad';}
  if(actKey==='sigmoid'&&sat>0.3){parts.push(`About ${Math.round(sat*100)}% of the final layer's sigmoid units are saturated near 0 or 1.`);cls='bad';}
  if(actKey==='relu'&&sat>0.9){parts.push(`About ${Math.round(sat*100)}% of the final ReLU units output exactly zero.`);cls=cls||'warn';}
  if(!isFinite(gIn)||gRatio>100){parts.push(`Going backward, the gradient grows about ${fmtSci(gRatio)}× by the time it reaches layer 1, so early layers would take wild, unstable steps.`);cls='bad';}
  else if(gRatio<0.01){parts.push(`Going backward, the gradient shrinks to about ${fmtSci(gRatio)} of its size by layer 1, so the first layers would barely learn.`);cls='bad';}
  else{parts.push(`The gradient at layer 1 is ${gRatio>=1?gRatio.toFixed(2)+"× larger than":(1/gRatio).toFixed(2)+"× smaller than"} the gradient at the top, which is close enough that every layer can learn.`);}
  if(cls===''&&actKey==='relu'&&(initKey==='xavierN'||initKey==='xavierU'||initKey==='lecun'))cls='warn';
  if(actKey==='relu'&&(initKey==='xavierN'||initKey==='xavierU'||initKey==='lecun')){parts.push("Notice the slow downward slope: ReLU drops half the signal's energy per layer and this scheme doesn't make up for it. Switch to He to flatten the line.");if(!cls)cls='warn';}
  if(actKey==='sigmoid'&&initKey!=='large')parts.push("With sigmoid, the gradient line slopes down whatever you choose, because its slope never exceeds 0.25.");
  if(cls==='')parts.push("This is the healthy picture good initialization aims for.");
  return{cls,text:parts.join(' ')};
}

/* The simulation runs in a Web Worker so big networks don't freeze the page.
   The worker source is assembled from the functions above; if workers are
   unavailable (some file:// setups) it falls back to the main thread. */
let l1Worker=null,l1Job=0;
const NL=String.fromCharCode(10);
function getWorker(){
  if(l1Worker!==null)return l1Worker||null;
  try{
    const fnTable=(o,keys)=>'{'+Object.keys(o).map(k=>k+':{'+keys.map(p=>p+':'+o[k][p].toString()).join(',')+'}').join(',')+'}';
    const src=[mulberry32,makeGauss,std,simulate].map(f=>f.toString()).join(NL)+
      NL+'const INITS='+fnTable(INITS,['fn'])+';'+NL+'const ACTS='+fnTable(ACTS,['f','d'])+';'+
      NL+'onmessage=e=>{const[id,...a]=e.data;const r=simulate(...a);postMessage({id,r},r.As.map(x=>x.buffer));};';
    l1Worker=new Worker(URL.createObjectURL(new Blob([src],{type:'text/javascript'})));
    l1Worker.onmessage=e=>{if(e.data.id!==l1Job)return;finishLab1(e.data.r);};
    l1Worker.onerror=()=>{l1Worker=false;runLab1();};
  }catch(err){l1Worker=false;}
  return l1Worker||null;
}
function finishLab1(res){
  l1Result=res;$('lab1').removeAttribute('aria-busy');renderLab1();updateTable(res.width);
}
function runLab1(){
  const depth=+$('l1-depth').value,width=+$('l1-width').value;
  $('l1-depth-o').textContent=depth;
  const args=[l1Init.value,l1Act.value,depth,width,l1Seed],w=getWorker();
  if(w){$('lab1').setAttribute('aria-busy','true');w.postMessage([++l1Job,...args]);}
  else finishLab1(simulate(...args));
}
function renderLab1(){
  const res=l1Result;if(!res)return;
  const strip=$('l1-strip');strip.innerHTML='';
  const range=ACTS[res.actKey].range;
  pickLayers(res.depth).forEach(l=>{
    const tile=document.createElement('div');tile.className='tile';
    const cv=document.createElement('canvas');tile.appendChild(cv);
    const cap=document.createElement('div');cap.className='cap';
    cap.innerHTML=`<span>Layer ${l+1}</span><span>σ <b>${fmtSci(res.actStd[l])}</b></span>`;
    tile.appendChild(cap);strip.appendChild(tile);
    drawHist(cv,res.As[l],range);
  });
  drawStdChart(res);
  const ins=insightFor(res);const el=$('l1-insight');el.textContent=ins.text;el.className='insight'+(ins.cls?' '+ins.cls:'');
}
function scheduleLab1(){clearTimeout(l1Timer);l1Timer=setTimeout(runLab1,40);}
[l1Init,l1Act,$('l1-width')].forEach(el=>el.addEventListener('change',scheduleLab1));
$('l1-depth').addEventListener('input',()=>{$('l1-depth-o').textContent=$('l1-depth').value;scheduleLab1();});
$('l1-resample').addEventListener('click',()=>{l1Seed=(l1Seed*7919+13)%100000;scheduleLab1();});
document.querySelectorAll('#lab1 .presets button').forEach(b=>b.addEventListener('click',()=>{const[i,a]=b.dataset.p.split(',');l1Init.value=i;l1Act.value=a;scheduleLab1();}));

/* ---------- Field guide table ---------- */
function updateTable(width){
  $('tbl-width-h').textContent=`Std at width ${width}`;
  const rows=['zeros','small','large','lecun','xavierU','xavierN','he'].map(k=>{
    const s=INITS[k];
    return `<tr><td>${s.label}</td><td class="f">${s.variance}</td><td>${s.built}</td><td>${s.fails}</td><td class="v">${s.sd(width).toFixed(4)}</td></tr>`;
  });
  $('tbl-body').innerHTML=rows.join('');
}

/* ======================================================
   LAB 2: training race
   ====================================================== */
const LR_STEPS=[0.01,0.03,0.05,0.1,0.2,0.3,0.5];
const MAX_STEPS=4000,BATCH=32,STEPS_PER_FRAME=6,GRID=44;
const RACE_INITS=['small','large','lecun','xavierU','xavierN','he','zeros'];
const laneDefaults=['small','large','xavierN','he'];
const laneColors=['--l1','--l2','--l3','--l4'];

function makeData(kind,seed){
  const r=mulberry32(seed),g=makeGauss(r),N=320,X=new Float64Array(N*2),Y=new Float64Array(N);
  for(let k=0;k<N;k++){
    let x,y,c;
    if(kind==='spiral'){c=k%2;const t=Math.floor(k/2)/(N/2);const ang=t*2.8*Math.PI+c*Math.PI;const rad=0.12+0.85*t;x=rad*Math.cos(ang)+g()*0.035;y=rad*Math.sin(ang)+g()*0.035;}
    else if(kind==='circles'){c=k%2;const ang=r()*2*Math.PI;const rad=c?r()*0.42:0.62+r()*0.33;x=rad*Math.cos(ang)+g()*0.02;y=rad*Math.sin(ang)+g()*0.02;}
    else{x=r()*2-1;y=r()*2-1;if(Math.abs(x)<0.08)x+=x<0?-0.08:0.08;if(Math.abs(y)<0.08)y+=y<0?-0.08:0.08;c=x*y>0?1:0;}
    X[2*k]=x;X[2*k+1]=y;Y[k]=c;
  }
  return{X,Y,N};
}
function sigm(z){return z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));}

class MLP{
  constructor(sizes,initKey,actKey,seed){
    this.sizes=sizes;this.act=ACTS[actKey];
    const rng=mulberry32(seed),g=makeGauss(rng),init=INITS[initKey].fn;
    this.W=[];this.b=[];
    for(let l=0;l<sizes.length-1;l++){const fi=sizes[l],fo=sizes[l+1];const W=new Float64Array(fi*fo);for(let i=0;i<W.length;i++)W[i]=init(fi,fo,g,rng);this.W.push(W);this.b.push(new Float64Array(fo));}
    this.gradRms=new Array(sizes.length-1).fill(0);
  }
  forward(X,B){
    const L=this.W.length;let a=X;const as=[X],zs=[];
    for(let l=0;l<L;l++){
      const fi=this.sizes[l],fo=this.sizes[l+1],W=this.W[l],bb=this.b[l],z=new Float64Array(B*fo);
      for(let b=0;b<B;b++){const ao=b*fi,zo=b*fo;for(let j=0;j<fo;j++){let s=bb[j];const wo=j*fi;for(let i=0;i<fi;i++)s+=a[ao+i]*W[wo+i];z[zo+j]=s;}}
      zs.push(z);
      if(l<L-1){const na=new Float64Array(z.length);for(let i=0;i<z.length;i++)na[i]=this.act.f(z[i]);a=na;as.push(na);}else a=z;
    }
    return{as,zs,out:a};
  }
  step(X,Y,B,lr){
    const{as,zs,out}=this.forward(X,B);const L=this.W.length;
    let loss=0;let dz=new Float64Array(B);
    for(let b=0;b<B;b++){const z=out[b],y=Y[b];loss+=Math.max(z,0)-z*y+Math.log1p(Math.exp(-Math.abs(z)));dz[b]=(sigm(z)-y)/B;}
    for(let l=L-1;l>=0;l--){
      const fi=this.sizes[l],fo=this.sizes[l+1],W=this.W[l],bb=this.b[l],ap=as[l];
      const gW=new Float64Array(fi*fo),gb=new Float64Array(fo);
      for(let b=0;b<B;b++){const ao=b*fi,zo=b*fo;for(let j=0;j<fo;j++){const d=dz[zo+j];if(d===0)continue;gb[j]+=d;const wo=j*fi;for(let i=0;i<fi;i++)gW[wo+i]+=d*ap[ao+i];}}
      let ss=0;for(let i=0;i<gW.length;i++)ss+=gW[i]*gW[i];this.gradRms[l]=0.9*this.gradRms[l]+0.1*Math.sqrt(ss/gW.length);
      if(l>0){
        const da=new Float64Array(B*fi);
        for(let b=0;b<B;b++){const ao=b*fi,zo=b*fo;for(let j=0;j<fo;j++){const d=dz[zo+j];if(d===0)continue;const wo=j*fi;for(let i=0;i<fi;i++)da[ao+i]+=d*W[wo+i];}}
        const zp=zs[l-1];for(let i=0;i<da.length;i++)da[i]*=this.act.d(zp[i]);dz=da;
      }
      for(let i=0;i<W.length;i++)W[i]-=lr*gW[i];for(let j=0;j<fo;j++)bb[j]-=lr*gb[j];
    }
    return loss/B;
  }
  evaluate(X,Y,N){
    const{out}=this.forward(X,N);let loss=0,correct=0;
    for(let b=0;b<N;b++){const z=out[b],y=Y[b];loss+=Math.max(z,0)-z*y+Math.log1p(Math.exp(-Math.abs(z)));if((z>0?1:0)===y)correct++;}
    return{loss:loss/N,acc:correct/N};
  }
}

const gridX=new Float64Array(GRID*GRID*2);
(function(){const lo=-1.15,hi=1.15;for(let r=0;r<GRID;r++)for(let c=0;c<GRID;c++){const k=r*GRID+c;gridX[2*k]=lo+(c+0.5)/GRID*(hi-lo);gridX[2*k+1]=hi-(r+0.5)/GRID*(hi-lo);}})();

let race=null,running=false,rafId=null;
const lanesEl=$('lanes');
const laneUI=laneDefaults.map((def,i)=>{
  const el=document.createElement('div');el.className='lane';el.style.setProperty('--c',`var(${laneColors[i]})`);
  const sel=document.createElement('select');sel.setAttribute('aria-label',`Initialization for network ${i+1}`);
  RACE_INITS.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=INITS[k].label;sel.appendChild(o);});
  sel.value=def;
  const bd=document.createElement('canvas');bd.className='bd';bd.setAttribute('role','img');bd.setAttribute('aria-label','Decision boundary');
  const stat=document.createElement('div');stat.className='stat';
  const status=document.createElement('div');status.className='status';
  const gb=document.createElement('canvas');gb.className='gb';gb.setAttribute('role','img');gb.setAttribute('aria-label','Gradient size per layer');
  const gl=document.createElement('div');gl.className='gl';gl.innerHTML='<span>input layer</span><span>gradient size</span><span>output</span>';
  el.append(sel,bd,stat,status,gb,gl);lanesEl.appendChild(el);
  sel.addEventListener('change',resetRace);
  return{el,sel,bd,stat,status,gb};
});

function resetRace(){
  running=false;if(rafId)cancelAnimationFrame(rafId);
  $('r-run').textContent='Start training';
  const depth=+$('r-depth').value,act=$('r-act').value,data=makeData($('r-data').value,11);
  const sizes=[2,...Array(depth).fill(24),1];
  race={data,step:0,lr:LR_STEPS[+$('r-lr').value],batchRng:mulberry32(99),
    lanes:laneUI.map(u=>({init:u.sel.value,net:new MLP(sizes,u.sel.value,act,2024),hist:[],loss:NaN,acc:NaN,dead:false,probs:null}))};
  race.lanes.forEach(L=>{const e=L.net.evaluate(data.X,data.Y,data.N);L.loss=e.loss;L.acc=e.acc;L.hist.push([0,e.loss]);computeBoundary(L);});
  renderLegend();renderRace();
  $('r-insight').textContent="Press start. Every network gets the same data, the same mini-batches and the same underlying random draws; only the scale of the starting weights differs. The dashed line marks a loss of ln 2 ≈ 0.693, which is what a network scores when it just guesses 50/50.";
  $('r-insight').className='insight';
}
function computeBoundary(L){
  if(L.dead)return;
  const{out}=L.net.forward(gridX,GRID*GRID);
  const p=new Float32Array(GRID*GRID);for(let i=0;i<p.length;i++)p[i]=sigm(out[i]);L.probs=p;
}
function trainFrame(){
  if(!running)return;
  const d=race.data,Xb=new Float64Array(BATCH*2),Yb=new Float64Array(BATCH);
  for(let s=0;s<STEPS_PER_FRAME&&race.step<MAX_STEPS;s++){
    for(let b=0;b<BATCH;b++){const k=Math.floor(race.batchRng()*d.N);Xb[2*b]=d.X[2*k];Xb[2*b+1]=d.X[2*k+1];Yb[b]=d.Y[k];}
    race.lanes.forEach(L=>{if(L.dead)return;const l=L.net.step(Xb,Yb,BATCH,race.lr);if(!isFinite(l))L.dead=true;});
    race.step++;
    if(race.step%20===0)race.lanes.forEach(L=>{if(L.dead)return;const e=L.net.evaluate(d.X,d.Y,d.N);if(!isFinite(e.loss)){L.dead=true;return;}L.loss=e.loss;L.acc=e.acc;L.hist.push([race.step,e.loss]);});
  }
  if(race.step%60===0||race.step>=MAX_STEPS)race.lanes.forEach(computeBoundary);
  renderRace();
  if(race.step>=MAX_STEPS){running=false;$('r-run').textContent='Finished';summarizeRace();return;}
  if(race.step%400===0)summarizeRace();
  rafId=requestAnimationFrame(trainFrame);
}

function renderLegend(){
  $('r-legend').innerHTML=race.lanes.map((L,i)=>`<span><span class="sw" style="background:var(${laneColors[i]})"></span>${INITS[L.init].label}</span>`).join('')+'<span><span class="sw" style="background:var(--muted);height:1px"></span>chance level (ln 2)</span>';
}

function drawBoundary(cv,L,data){
  const{ctx,w,h}=setupCanvas(cv,cv.clientWidth||200);
  ctx.clearRect(0,0,w,h);
  const c0=hexToRgb(cssv('--k0')),c1=hexToRgb(cssv('--k1')),bg=hexToRgb(cssv('--surface'));
  if(L.probs&&!L.dead){
    const off=document.createElement('canvas');off.width=GRID;off.height=GRID;const octx=off.getContext('2d');
    const img=octx.createImageData(GRID,GRID);
    for(let i=0;i<GRID*GRID;i++){
      const p=L.probs[i],conf=Math.abs(p-0.5)*2,a=0.12+0.38*conf,col=p>0.5?c1:c0;
      img.data[4*i]=Math.round(bg[0]*(1-a)+col[0]*a);img.data[4*i+1]=Math.round(bg[1]*(1-a)+col[1]*a);img.data[4*i+2]=Math.round(bg[2]*(1-a)+col[2]*a);img.data[4*i+3]=255;
    }
    octx.putImageData(img,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(off,0,0,w,h);
  }
  const map=v=>(v+1.15)/2.3;
  for(let k=0;k<data.N;k++){
    const x=map(data.X[2*k])*w,y=(1-map(data.X[2*k+1]))*h;
    ctx.beginPath();ctx.arc(x,y,2.2,0,Math.PI*2);ctx.fillStyle=data.Y[k]?cssv('--k1'):cssv('--k0');ctx.fill();
  }
  if(L.dead){ctx.fillStyle='rgba(0,0,0,0.35)';ctx.fillRect(0,0,w,h);ctx.fillStyle='#fff';ctx.font="600 15px 'Segoe UI', system-ui, sans-serif";ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('Diverged',w/2,h/2);}
}
function drawGradBars(cv,L,color){
  const{ctx,w,h}=setupCanvas(cv,46);ctx.clearRect(0,0,w,h);
  const g=L.net.gradRms,n=g.length,gap=3,bw=(w-gap*(n-1))/n;
  const lo=-9,hi=0;
  ctx.fillStyle=cssv('--surface');ctx.fillRect(0,0,w,h);
  for(let i=0;i<n;i++){
    let v=g[i];let t;
    if(!isFinite(v))t=1;else if(v<=0)t=0;else t=Math.min(1,Math.max(0,(Math.log10(v)-lo)/(hi-lo)));
    ctx.fillStyle=color;ctx.globalAlpha=0.9;ctx.fillRect(i*(bw+gap),h-t*h,bw,t*h);ctx.globalAlpha=1;
  }
}
function renderRace(){
  if(!race)return;
  $('r-step').textContent=`Step ${race.step} of ${MAX_STEPS}`;
  race.lanes.forEach((L,i)=>{
    const u=laneUI[i],col=cssv(laneColors[i]);
    drawBoundary(u.bd,L,race.data);
    drawGradBars(u.gb,L,col);
    u.stat.innerHTML=L.dead?'<span>Loss <b>NaN</b></span><span>Acc <b>—</b></span>':`<span>Loss <b>${L.loss.toFixed(3)}</b></span><span>Acc <b>${Math.round(L.acc*100)}%</b></span>`;
    let s='';
    if(L.dead)s='Weights overflowed to infinity.';
    else if(race.step===0)s='Untrained.';
    else if(race.step>=800&&Math.abs(L.loss-Math.LN2)<0.02)s='Stuck at chance: gradients too small to move.';
    else if(L.acc>=0.97)s='Solved.';
    else if(race.step>=800&&L.loss>0.6)s='Barely learning.';
    else s='Learning…';
    u.status.textContent=s;
  });
  drawLossChart();
}
function drawLossChart(){
  const cv=$('r-loss');const{ctx,w,h}=setupCanvas(cv,window.innerWidth<600?190:230);
  ctx.clearRect(0,0,w,h);
  const padL=40,padR=12,padT=10,padB=26,pw=w-padL-padR,ph=h-padT-padB,ymax=1.2;
  const X=s=>padL+s/MAX_STEPS*pw,Y=v=>padT+ph-Math.min(ymax,v)/ymax*ph;
  ctx.font=displayFont;ctx.fillStyle=cssv('--muted');ctx.strokeStyle=cssv('--line');ctx.lineWidth=1;ctx.textAlign='right';ctx.textBaseline='middle';
  [0,0.4,0.8,1.2].forEach(v=>{const y=Y(v);ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(w-padR,y);ctx.stroke();ctx.fillText(v.toFixed(1),padL-6,y);});
  ctx.textAlign='center';ctx.textBaseline='top';[0,1000,2000,3000,4000].forEach(s=>ctx.fillText(s===0?'0':s/1000+'k',X(s),h-padB+7));
  ctx.textAlign='left';ctx.fillText('loss',padL+4,padT);
  ctx.setLineDash([5,4]);ctx.strokeStyle=cssv('--muted');ctx.beginPath();ctx.moveTo(padL,Y(Math.LN2));ctx.lineTo(w-padR,Y(Math.LN2));ctx.stroke();ctx.setLineDash([]);
  race.lanes.forEach((L,i)=>{
    ctx.strokeStyle=cssv(laneColors[i]);ctx.lineWidth=2.2;ctx.beginPath();
    L.hist.forEach(([s,v],k)=>{const x=X(s),y=Y(v);k?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    if(L.dead&&L.hist.length){const[s,v]=L.hist[L.hist.length-1];ctx.fillStyle=cssv(laneColors[i]);ctx.font="600 13px 'Segoe UI', system-ui, sans-serif";ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillText('✕',X(s)+2,Y(v));ctx.font=displayFont;}
  });
}
function summarizeRace(){
  const ranked=race.lanes.map((L,i)=>({L,i})).filter(o=>!o.L.dead).sort((a,b)=>a.L.loss-b.L.loss);
  const dead=race.lanes.filter(L=>L.dead).map(L=>INITS[L.init].label);
  const stuck=race.lanes.filter(L=>!L.dead&&Math.abs(L.loss-Math.LN2)<0.02).map(L=>INITS[L.init].label);
  const parts=[];
  if(ranked.length)parts.push(`After ${race.step} steps, ${INITS[ranked[0].L.init].label} leads with a loss of ${ranked[0].L.loss.toFixed(3)} and ${Math.round(ranked[0].L.acc*100)}% accuracy.`);
  if(stuck.length)parts.push(`${stuck.join(' and ')} ${stuck.length>1?'are':'is'} still stuck at chance level: look at ${stuck.length>1?'their':'its'} gradient bars, which are nearly flat on the input side because the gradient vanished on the way down.`);
  if(dead.length)parts.push(`${dead.join(' and ')} diverged: the weights started so large that the first updates overshot and blew up.`);
  const big=race.lanes.find(L=>L.init==='large'&&!L.dead&&L.loss<0.15);
  if(big&&$('r-act').value!=='relu')parts.push('Surprised that σ = 1 does well here? With only two inputs the first layer is not oversized, and saturated units behave like sharp step functions that can still carve up a simple 2D dataset. The saturation problem bites much harder with many inputs, as the signal lab above shows.');
  if(!stuck.length&&!dead.length&&!big)parts.push('Try more hidden layers, or switch to ReLU or sigmoid, to see the gap between schemes widen.');
  $('r-insight').textContent=parts.join(' ');
  $('r-insight').className='insight'+(stuck.length||dead.length?' warn':'');
}

$('r-run').addEventListener('click',()=>{
  if(race.step>=MAX_STEPS){resetRace();}
  running=!running;$('r-run').textContent=running?'Pause':(race.step>0?'Resume':'Start training');
  if(running)rafId=requestAnimationFrame(trainFrame);
});
$('r-reset').addEventListener('click',resetRace);
['r-data','r-act'].forEach(id=>$(id).addEventListener('change',resetRace));
$('r-depth').addEventListener('input',()=>{$('r-depth-o').textContent=$('r-depth').value;resetRace();});
$('r-lr').addEventListener('input',()=>{$('r-lr-o').textContent=LR_STEPS[+$('r-lr').value];resetRace();});

/* ---------- Redraw on resize / theme change ---------- */
let rzTimer=null;
window.addEventListener('resize',()=>{clearTimeout(rzTimer);rzTimer=setTimeout(()=>{renderLab1();renderRace();},120);});
try{window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{renderLab1();renderRace();});}catch(e){}
new MutationObserver(()=>{renderLab1();renderRace();}).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});

/* ---------- Remember settings between visits ---------- */
const SAVE_KEY='weight-init-lab:v1';
const saved=[['l1-init',l1Init],['l1-act',l1Act],['l1-depth',$('l1-depth')],['l1-width',$('l1-width')],
  ['r-data',$('r-data')],['r-act',$('r-act')],['r-depth',$('r-depth')],['r-lr',$('r-lr')],
  ...laneUI.map((u,i)=>['lane'+i,u.sel])];
function loadSettings(){
  try{const d=JSON.parse(localStorage.getItem(SAVE_KEY)||'{}');
    saved.forEach(([k,el])=>{if(d[k]!==undefined&&[...(el.options||[1])].some(o=>o===1||o.value===d[k]))el.value=d[k];});
    $('r-depth-o').textContent=$('r-depth').value;$('r-lr-o').textContent=LR_STEPS[+$('r-lr').value];
  }catch(e){}
}
function saveSettings(){try{const d={};saved.forEach(([k,el])=>d[k]=el.value);localStorage.setItem(SAVE_KEY,JSON.stringify(d));}catch(e){}}
document.addEventListener('change',saveSettings);document.addEventListener('input',saveSettings);

function start(){loadSettings();runLab1();resetRace();}
if(document.fonts&&document.fonts.ready){document.fonts.ready.then(start).catch(start);}else start();
})();
