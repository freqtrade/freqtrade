import{_ as te}from"./DraggableContainer.vue_vue_type_script_setup_true_lang-CjD_EOpm.js";import{_ as ne}from"./ExchangeSelect.vue_vue_type_script_setup_true_lang-BAa3CZVG.js";import{s as se}from"./index-C6-JHvJO.js";import{$ as ae,ab as re,ag as oe,c as i,a as s,k as y,O as C,a2 as le,x as g,z as S,I as ie,e as t,d as F,c7 as de,l as v,F as U,m as E,b as l,f as n,g as I,h as c,B as ue,j as me,r as _,u as ce,S as pe,U as ge,i as T,v as fe,X as _e,a3 as ve,Z as N,W as O,c1 as be,Y as A,t as xe}from"./index-CUXw-u_C.js";import{u as ye,_ as he,a as ke}from"./pairlistConfig-DmEHj1or.js";import{s as we}from"./index-DL2TUmVu.js";import{_ as Se}from"./TimeRangeSelect.vue_vue_type_script_setup_true_lang-KFM3m_7G.js";import{_ as Ve}from"./check-C21h_Umc.js";import"./plus-box-outline-1BbsrXsa.js";var Te=`
    .p-progressbar {
        display: block;
        position: relative;
        overflow: hidden;
        height: dt('progressbar.height');
        background: dt('progressbar.background');
        border-radius: dt('progressbar.border.radius');
    }

    .p-progressbar-value {
        margin: 0;
        background: dt('progressbar.value.background');
    }

    .p-progressbar-label {
        color: dt('progressbar.label.color');
        font-size: dt('progressbar.label.font.size');
        font-weight: dt('progressbar.label.font.weight');
    }

    .p-progressbar-determinate .p-progressbar-value {
        height: 100%;
        width: 0%;
        position: absolute;
        display: none;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        transition: width 1s ease-in-out;
    }

    .p-progressbar-determinate .p-progressbar-label {
        display: inline-flex;
    }

    .p-progressbar-indeterminate .p-progressbar-value::before {
        content: '';
        position: absolute;
        background: inherit;
        inset-block-start: 0;
        inset-inline-start: 0;
        inset-block-end: 0;
        will-change: inset-inline-start, inset-inline-end;
        animation: p-progressbar-indeterminate-anim 2.1s cubic-bezier(0.65, 0.815, 0.735, 0.395) infinite;
    }

    .p-progressbar-indeterminate .p-progressbar-value::after {
        content: '';
        position: absolute;
        background: inherit;
        inset-block-start: 0;
        inset-inline-start: 0;
        inset-block-end: 0;
        will-change: inset-inline-start, inset-inline-end;
        animation: p-progressbar-indeterminate-anim-short 2.1s cubic-bezier(0.165, 0.84, 0.44, 1) infinite;
        animation-delay: 1.15s;
    }

    @keyframes p-progressbar-indeterminate-anim {
        0% {
            inset-inline-start: -35%;
            inset-inline-end: 100%;
        }
        60% {
            inset-inline-start: 100%;
            inset-inline-end: -90%;
        }
        100% {
            inset-inline-start: 100%;
            inset-inline-end: -90%;
        }
    }
    @-webkit-keyframes p-progressbar-indeterminate-anim {
        0% {
            inset-inline-start: -35%;
            inset-inline-end: 100%;
        }
        60% {
            inset-inline-start: 100%;
            inset-inline-end: -90%;
        }
        100% {
            inset-inline-start: 100%;
            inset-inline-end: -90%;
        }
    }

    @keyframes p-progressbar-indeterminate-anim-short {
        0% {
            inset-inline-start: -200%;
            inset-inline-end: 100%;
        }
        60% {
            inset-inline-start: 107%;
            inset-inline-end: -8%;
        }
        100% {
            inset-inline-start: 107%;
            inset-inline-end: -8%;
        }
    }
    @-webkit-keyframes p-progressbar-indeterminate-anim-short {
        0% {
            inset-inline-start: -200%;
            inset-inline-end: 100%;
        }
        60% {
            inset-inline-start: 107%;
            inset-inline-end: -8%;
        }
        100% {
            inset-inline-start: 107%;
            inset-inline-end: -8%;
        }
    }
`,$e={root:function(u){var f=u.instance;return["p-progressbar p-component",{"p-progressbar-determinate":f.determinate,"p-progressbar-indeterminate":f.indeterminate}]},value:"p-progressbar-value",label:"p-progressbar-label"},De=ae.extend({name:"progressbar",style:Te,classes:$e}),Ce={name:"BaseProgressBar",extends:re,props:{value:{type:Number,default:null},mode:{type:String,default:"determinate"},showValue:{type:Boolean,default:!0}},style:De,provide:function(){return{$pcProgressBar:this,$parentInstance:this}}},J={name:"ProgressBar",extends:Ce,inheritAttrs:!1,computed:{progressStyle:function(){return{width:this.value+"%",display:"flex"}},indeterminate:function(){return this.mode==="indeterminate"},determinate:function(){return this.mode==="determinate"},dataP:function(){return oe({determinate:this.determinate,indeterminate:this.indeterminate})}}},Be=["aria-valuenow","data-p"],Pe=["data-p"],Ue=["data-p"],Ee=["data-p"];function Me(a,u,f,b,h,o){return s(),i("div",C({role:"progressbar",class:a.cx("root"),"aria-valuemin":"0","aria-valuenow":a.value,"aria-valuemax":"100","data-p":o.dataP},a.ptmi("root")),[o.determinate?(s(),i("div",C({key:0,class:a.cx("value"),style:o.progressStyle,"data-p":o.dataP},a.ptm("value")),[a.value!=null&&a.value!==0&&a.showValue?(s(),i("div",C({key:0,class:a.cx("label"),"data-p":o.dataP},a.ptm("label")),[le(a.$slots,"default",{},function(){return[g(S(a.value+"%"),1)]})],16,Ue)):y("",!0)],16,Pe)):o.indeterminate?(s(),i("div",C({key:1,class:a.cx("value"),"data-p":o.dataP},a.ptm("value")),null,16,Ee)):y("",!0)],16,Be)}J.render=Me;const ze={viewBox:"0 0 24 24",width:"1.2em",height:"1.2em"};function Ne(a,u){return s(),i("svg",ze,[...u[0]||(u[0]=[t("path",{fill:"currentColor",d:"M8 17v-2h8v2zm8-7l-4 4l-4-4h2.5V7h3v3zM5 3h14a2 2 0 0 1 2 2v14c0 1.11-.89 2-2 2H5a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2m0 2v14h14V5z"},null,-1)])])}const Oe=ie({name:"mdi-download-box-outline",render:Ne}),Ae={class:"flex flex-row items-end gap-1"},Fe={class:"ms-2 w-full grow space-y-1"},Ie=["title"],Je={key:1},Re={class:"flex justify-between"},He={key:1},Le={key:2,class:"w-25"},qe={key:3,class:"flex flex-col md:flex-row w-full grow gap-2"},We=F({__name:"BackgroundJobTracking",setup(a){const{runningJobs:u,clearJobs:f}=de();return(b,h)=>{const o=Oe,B=Ve,p=J,k=ue,w=I;return s(),i("div",Ae,[t("ul",Fe,[(s(!0),i(U,null,E(n(u),(d,x)=>(s(),i("li",{key:x,class:"border p-1 pb-2 rounded-sm dark:border-surface-700 border-surface-300 flex gap-2 items-center",title:x},[d.taskStatus?.job_category==="download_data"?(s(),v(o,{key:0})):(s(),i("span",Je,S(d.taskStatus?.job_category),1)),t("div",Re,[d.taskStatus?.status==="success"?(s(),v(B,{key:0,class:"text-success",title:""})):(s(),i("span",He,S(d.taskStatus?.status),1)),d.taskStatus?.progress?(s(),i("span",Le,S(d.taskStatus?.progress),1)):y("",!0)]),d.taskStatus?.progress?(s(),v(p,{key:2,class:"w-full grow",value:d.taskStatus?.progress/100*100,"show-progress":"",max:100,striped:""},null,8,["value"])):y("",!0),d.taskStatus?.progress_tasks?(s(),i("div",qe,[(s(!0),i(U,null,E(Object.entries(d.taskStatus?.progress_tasks),([P,V])=>(s(),i("div",{key:P,class:"w-full"},[g(S(V.description)+" ",1),l(p,{class:"w-full grow",value:Math.round(V.progress/V.total*100*100)/100,"show-progress":"",pt:{value:{class:d.taskStatus.status==="success"?"bg-emerald-500":"bg-amber-500"}},striped:""},null,8,["value","pt"])]))),128))])):y("",!0)],8,Ie))),128))]),Object.keys(n(u)).length>0?(s(),v(w,{key:0,severity:"secondary",class:"ms-auto",onClick:n(f)},{icon:c(()=>[l(k)]),_:1},8,["onClick"])):y("",!0)])}}}),Xe=_([{description:"All USDT Pairs",pairs:[".*/USDT"]},{description:"All USDT Futures Pairs",pairs:[".*/USDT:USDT"]}]);function Ye(){return{pairTemplates:me(()=>Xe.value.map((a,u)=>({...a,idx:u})))}}const Ze={class:"px-1 mx-auto w-full max-w-4xl lg:max-w-7xl"},Ge={class:"flex mb-3 gap-3 flex-col"},Ke={class:"flex flex-col gap-3"},Qe={class:"flex flex-col lg:flex-row gap-3"},je={class:"flex-fill"},et={class:"flex flex-col gap-2"},tt={class:"flex gap-2"},nt={class:"flex flex-col gap-1"},st={class:"flex flex-col gap-1"},at={class:"flex-fill px-3"},rt={class:"flex flex-col gap-2"},ot={class:"px-3 border dark:border-surface-700 border-surface-300 p-2 rounded-sm"},lt={class:"flex flex-col gap-2"},it={class:"flex justify-between items-center"},dt={key:0},ut={key:1,class:"flex items-center gap-2"},mt={class:"mb-2 border dark:border-surface-700 border-surface-300 rounded-sm p-2 text-start"},ct={class:"mb-2 border dark:border-surface-700 border-surface-300 rounded-md p-2 text-start"},pt={class:"grid grid-cols md:grid-cols-2 items-center gap-2"},gt={class:"mb-2 border dark:border-surface-700 border-surface-300 rounded-md p-2 text-start"},ft={class:"px-3"},_t=F({__name:"DownloadDataMain",setup(a){const u=ce(),f=ye(),b=_(["BTC/USDT","ETH/USDT",""]),h=_(["5m","1h"]),o=_({useCustomTimerange:!1,timerange:"",days:30}),{pairTemplates:B}=Ye(),p=_({customExchange:!1,selectedExchange:{exchange:"binance",trade_mode:{margin_mode:ge.NONE,trading_mode:pe.SPOT}}}),k=_(!1),w=_(!1),d=_(!1),x=_([]),P=[{text:"Spot",value:"spot"},{text:"Futures",value:"futures"},{text:"Funding Rate",value:"funding_rate"},{text:"Mark",value:"mark"},{text:"Index",value:"index"},{text:"Premium Index",value:"premiumIndex"}];function V(m){b.value.push(...m)}function R(m){b.value=[...m]}async function H(){const m={pairs:b.value.filter(e=>e!==""),timeframes:h.value.filter(e=>e!=="")};o.value.useCustomTimerange&&o.value.timerange?m.timerange=o.value.timerange:m.days=o.value.days,d.value&&(m.erase=k.value,m.download_trades=w.value,p.value.customExchange&&(m.exchange=p.value.selectedExchange.exchange,m.trading_mode=p.value.selectedExchange.trade_mode.trading_mode,m.margin_mode=p.value.selectedExchange.trade_mode.margin_mode),u.activeBot.botFeatures.downloadDataCandleTypes&&x.value.length>0&&(m.candle_types=x.value)),await u.activeBot.startDataDownload(m)}return(m,e)=>{const L=We,M=he,$=I,q=fe,D=_e,W=Se,X=we,Y=ve,Z=ke,G=be,K=se,Q=ne,j=te;return s(),i("div",Ze,[l(L,{class:"mb-4"}),l(j,{header:"Downloading Data",class:"mx-1 p-4"},{default:c(()=>[t("div",Ge,[t("div",Ke,[t("div",Qe,[t("div",je,[t("div",et,[e[13]||(e[13]=t("div",{class:"flex justify-between"},[t("h4",{class:"text-start font-bold text-lg"},"Select Pairs"),t("h5",{class:"text-start font-bold text-lg"},"Pairs from template")],-1)),t("div",tt,[l(M,{modelValue:n(b),"onUpdate:modelValue":e[0]||(e[0]=r=>T(b)?b.value=r:null),placeholder:"Pair",size:"small",class:"grow"},null,8,["modelValue"]),t("div",nt,[t("div",st,[(s(!0),i(U,null,E(n(B),r=>(s(),v($,{key:r.idx,severity:"secondary",title:r.pairs.reduce((z,ee)=>`${z}${ee}
`,""),onClick:z=>V(r.pairs)},{default:c(()=>[g(S(r.description),1)]),_:2},1032,["title","onClick"]))),128))]),l(q),l($,{disabled:n(f).whitelist.length===0,title:"Add all pairs from Pairlist Config - requires the pairlist config to have ran first.",severity:"secondary",onClick:e[1]||(e[1]=r=>R(n(f).whitelist))},{default:c(()=>[...e[12]||(e[12]=[g(" Use Pairs from Pairlist Config ",-1)])]),_:1},8,["disabled"])])])])]),t("div",at,[t("div",rt,[e[14]||(e[14]=t("h4",{class:"text-start font-bold text-lg"},"Select timeframes",-1)),l(M,{modelValue:n(h),"onUpdate:modelValue":e[2]||(e[2]=r=>T(h)?h.value=r:null),placeholder:"Timeframe"},null,8,["modelValue"])])])]),t("div",ot,[t("div",lt,[t("div",it,[e[16]||(e[16]=t("h4",{class:"text-start mb-0 font-bold text-lg"},"Time Selection",-1)),l(D,{modelValue:n(o).useCustomTimerange,"onUpdate:modelValue":e[3]||(e[3]=r=>n(o).useCustomTimerange=r),class:"mb-0",switch:""},{default:c(()=>[...e[15]||(e[15]=[g(" Use custom timerange ",-1)])]),_:1},8,["modelValue"])]),n(o).useCustomTimerange?(s(),i("div",dt,[l(W,{modelValue:n(o).timerange,"onUpdate:modelValue":e[4]||(e[4]=r=>n(o).timerange=r)},null,8,["modelValue"])])):(s(),i("div",ut,[e[17]||(e[17]=t("label",null,"Days to download:",-1)),l(X,{modelValue:n(o).days,"onUpdate:modelValue":e[5]||(e[5]=r=>n(o).days=r),type:"number","aria-label":"Days to download",min:1,step:1,size:"small"},null,8,["modelValue"])]))])]),t("div",mt,[l($,{class:"mb-2",severity:"secondary",onClick:e[6]||(e[6]=r=>d.value=!n(d))},{default:c(()=>[e[18]||(e[18]=g(" Advanced Options ",-1)),n(d)?(s(),v(Z,{key:1})):(s(),v(Y,{key:0}))]),_:1}),l(N,null,{default:c(()=>[O(t("div",null,[l(G,{severity:"info",class:"mb-2 py-2"},{default:c(()=>[...e[19]||(e[19]=[g(" Advanced options (Erase data, Download trades, and Custom Exchange settings) will only be applied when this section is expanded. ",-1)])]),_:1}),t("div",ct,[l(D,{modelValue:n(k),"onUpdate:modelValue":e[7]||(e[7]=r=>T(k)?k.value=r:null),class:"mb-2"},{default:c(()=>[...e[20]||(e[20]=[g("Erase existing data",-1)])]),_:1},8,["modelValue"]),l(D,{modelValue:n(w),"onUpdate:modelValue":e[8]||(e[8]=r=>T(w)?w.value=r:null),class:"mb-2"},{default:c(()=>[...e[21]||(e[21]=[g(" Download Trades instead of OHLCV data ",-1)])]),_:1},8,["modelValue"]),t("div",pt,[n(u).activeBot.botFeatures.downloadDataCandleTypes?(s(),v(K,{key:0,modelValue:n(x),"onUpdate:modelValue":e[9]||(e[9]=r=>T(x)?x.value=r:null),options:P,"option-label":"text","option-value":"value",placeholder:"Select Candle Types"},null,8,["modelValue"])):y("",!0),e[22]||(e[22]=t("small",null,"When no candle-type is selected, freqtrade will download the necessary candle types for regular operation automatically.",-1))])]),t("div",gt,[l(D,{modelValue:n(p).customExchange,"onUpdate:modelValue":e[10]||(e[10]=r=>n(p).customExchange=r),class:"mb-2"},{default:c(()=>[...e[23]||(e[23]=[g(" Custom Exchange ",-1)])]),_:1},8,["modelValue"]),l(N,{name:"fade"},{default:c(()=>[O(l(Q,{modelValue:n(p).selectedExchange,"onUpdate:modelValue":e[11]||(e[11]=r=>n(p).selectedExchange=r)},null,8,["modelValue"]),[[A,n(p).customExchange]])]),_:1})])],512),[[A,n(d)]])]),_:1})]),t("div",ft,[l($,{severity:"primary",onClick:H},{default:c(()=>[...e[24]||(e[24]=[g("Start Download",-1)])]),_:1})])])])]),_:1})])}}}),vt={};function bt(a,u){const f=_t;return s(),v(f,{class:"pt-4"})}const Dt=xe(vt,[["render",bt]]);export{Dt as default};
//# sourceMappingURL=DownloadDataView-vVwFA2rr.js.map
