import{$ as b,ab as y,c as i,a as o,a2 as $,O as m,I as E,e as f,d as P,r as _,o as M,V as H,w as F,l as v,f as c,s as O,i as R,n as W,k as g,F as x,b as p,g as j,h as d,a9 as T,B as U,_ as q}from"./index-CUXw-u_C.js";import{_ as J}from"./check-C21h_Umc.js";import{_ as K}from"./plus-box-outline-1BbsrXsa.js";var L=`
    .p-inputgroup,
    .p-inputgroup .p-iconfield,
    .p-inputgroup .p-floatlabel,
    .p-inputgroup .p-iftalabel {
        display: flex;
        align-items: stretch;
        width: 100%;
    }

    .p-inputgroup .p-inputtext,
    .p-inputgroup .p-inputwrapper {
        flex: 1 1 auto;
        width: 1%;
    }

    .p-inputgroupaddon {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: dt('inputgroup.addon.padding');
        background: dt('inputgroup.addon.background');
        color: dt('inputgroup.addon.color');
        border-block-start: 1px solid dt('inputgroup.addon.border.color');
        border-block-end: 1px solid dt('inputgroup.addon.border.color');
        min-width: dt('inputgroup.addon.min.width');
    }

    .p-inputgroupaddon:first-child,
    .p-inputgroupaddon + .p-inputgroupaddon {
        border-inline-start: 1px solid dt('inputgroup.addon.border.color');
    }

    .p-inputgroupaddon:last-child {
        border-inline-end: 1px solid dt('inputgroup.addon.border.color');
    }

    .p-inputgroupaddon:has(.p-button) {
        padding: 0;
        overflow: hidden;
    }

    .p-inputgroupaddon .p-button {
        border-radius: 0;
    }

    .p-inputgroup > .p-component,
    .p-inputgroup > .p-inputwrapper > .p-component,
    .p-inputgroup > .p-iconfield > .p-component,
    .p-inputgroup > .p-floatlabel > .p-component,
    .p-inputgroup > .p-floatlabel > .p-inputwrapper > .p-component,
    .p-inputgroup > .p-iftalabel > .p-component,
    .p-inputgroup > .p-iftalabel > .p-inputwrapper > .p-component {
        border-radius: 0;
        margin: 0;
    }

    .p-inputgroupaddon:first-child,
    .p-inputgroup > .p-component:first-child,
    .p-inputgroup > .p-inputwrapper:first-child > .p-component,
    .p-inputgroup > .p-iconfield:first-child > .p-component,
    .p-inputgroup > .p-floatlabel:first-child > .p-component,
    .p-inputgroup > .p-floatlabel:first-child > .p-inputwrapper > .p-component,
    .p-inputgroup > .p-iftalabel:first-child > .p-component,
    .p-inputgroup > .p-iftalabel:first-child > .p-inputwrapper > .p-component {
        border-start-start-radius: dt('inputgroup.addon.border.radius');
        border-end-start-radius: dt('inputgroup.addon.border.radius');
    }

    .p-inputgroupaddon:last-child,
    .p-inputgroup > .p-component:last-child,
    .p-inputgroup > .p-inputwrapper:last-child > .p-component,
    .p-inputgroup > .p-iconfield:last-child > .p-component,
    .p-inputgroup > .p-floatlabel:last-child > .p-component,
    .p-inputgroup > .p-floatlabel:last-child > .p-inputwrapper > .p-component,
    .p-inputgroup > .p-iftalabel:last-child > .p-component,
    .p-inputgroup > .p-iftalabel:last-child > .p-inputwrapper > .p-component {
        border-start-end-radius: dt('inputgroup.addon.border.radius');
        border-end-end-radius: dt('inputgroup.addon.border.radius');
    }

    .p-inputgroup .p-component:focus,
    .p-inputgroup .p-component.p-focus,
    .p-inputgroup .p-inputwrapper-focus,
    .p-inputgroup .p-component:focus ~ label,
    .p-inputgroup .p-component.p-focus ~ label,
    .p-inputgroup .p-inputwrapper-focus ~ label {
        z-index: 1;
    }

    .p-inputgroup > .p-button:not(.p-button-icon-only) {
        width: auto;
    }

    .p-inputgroup .p-iconfield + .p-iconfield .p-inputtext {
        border-inline-start: 0;
    }
`,Q={root:"p-inputgroup"},X=b.extend({name:"inputgroup",style:L,classes:Q}),Y={name:"BaseInputGroup",extends:y,style:X,provide:function(){return{$pcInputGroup:this,$parentInstance:this}}},Z={name:"InputGroup",extends:Y,inheritAttrs:!1};function nn(n,a,r,s,e,t){return o(),i("div",m({class:n.cx("root")},n.ptmi("root")),[$(n.$slots,"default")],16)}Z.render=nn;var en={root:"p-inputgroupaddon"},tn=b.extend({name:"inputgroupaddon",classes:en}),on={name:"BaseInputGroupAddon",extends:y,style:tn,provide:function(){return{$pcInputGroupAddon:this,$parentInstance:this}}},pn={name:"InputGroupAddon",extends:on,inheritAttrs:!1};function rn(n,a,r,s,e,t){return o(),i("div",m({class:n.cx("root")},n.ptmi("root")),[$(n.$slots,"default")],16)}pn.render=rn;const an={viewBox:"0 0 24 24",width:"1.2em",height:"1.2em"};function sn(n,a){return o(),i("svg",an,[...a[0]||(a[0]=[f("path",{fill:"currentColor",d:"M19 21H8V7h11m0-2H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2m-3-4H4a2 2 0 0 0-2 2v14h2V3h12z"},null,-1)])])}const ln=E({name:"mdi-content-copy",render:sn}),un={class:"grow"},kn=P({__name:"EditValue",props:{modelValue:{},allowEdit:{type:Boolean,default:!1},allowAdd:{type:Boolean,default:!1},allowDuplicate:{type:Boolean,default:!1},editableName:{},alignVertical:{type:Boolean,default:!1}},emits:["delete","new","duplicate","rename"],setup(n,{emit:a}){const r=n,s=a,e=_(""),t=_(0);M(()=>{e.value=r.modelValue});function V(){t.value=0,e.value=r.modelValue}function B(){e.value=e.value+" (copy)",t.value=3}function S(){e.value="",t.value=2}H(()=>r.modelValue,()=>{e.value=r.modelValue});function k(){t.value===2?s("new",e.value):t.value===3?s("duplicate",r.modelValue,e.value):s("rename",r.modelValue,e.value),t.value=0}return(w,l)=>{const I=O,N=T,u=j,A=ln,C=U,z=K,G=J,D=q;return o(),i("form",{class:"flex flex-row",onSubmit:F(k,["prevent"])},[f("div",un,[c(t)===0?$(w.$slots,"default",{key:0}):(o(),v(I,{key:1,modelValue:c(e),"onUpdate:modelValue":l[0]||(l[0]=h=>R(e)?e.value=h:null),size:"small",fluid:""},null,8,["modelValue"]))]),f("div",{class:W(["mt-auto flex gap-1 ms-1",n.alignVertical?"flex-col":"flex-row"])},[n.allowEdit&&c(t)===0?(o(),i(x,{key:0},[p(u,{size:"small",severity:"secondary",title:`Edit this ${n.editableName}.`,onClick:l[1]||(l[1]=h=>t.value=1)},{icon:d(()=>[p(N)]),_:1},8,["title"]),n.allowDuplicate?(o(),v(u,{key:0,size:"small",severity:"secondary",title:`Duplicate ${n.editableName}.`,onClick:B},{icon:d(()=>[p(A)]),_:1},8,["title"])):g("",!0),p(u,{size:"small",severity:"secondary",title:`Delete this ${n.editableName}.`,onClick:l[2]||(l[2]=h=>w.$emit("delete",n.modelValue))},{icon:d(()=>[p(C)]),_:1},8,["title"])],64)):g("",!0),n.allowAdd&&c(t)===0?(o(),v(u,{key:1,size:"small",title:`Add new ${n.editableName}.`,severity:"primary",onClick:S},{icon:d(()=>[p(z)]),_:1},8,["title"])):g("",!0),c(t)!==0?(o(),i(x,{key:2},[p(u,{size:"small",title:`Add new ${n.editableName}`,severity:"primary",onClick:k},{icon:d(()=>[p(G)]),_:1},8,["title"]),p(u,{size:"small",title:"Abort",severity:"secondary",onClick:V},{icon:d(()=>[p(D)]),_:1})],64)):g("",!0)],2)],32)}}});var dn=`
    .p-progressspinner {
        position: relative;
        margin: 0 auto;
        width: 100px;
        height: 100px;
        display: inline-block;
    }

    .p-progressspinner::before {
        content: '';
        display: block;
        padding-top: 100%;
    }

    .p-progressspinner-spin {
        height: 100%;
        transform-origin: center center;
        width: 100%;
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        right: 0;
        margin: auto;
        animation: p-progressspinner-rotate 2s linear infinite;
    }

    .p-progressspinner-circle {
        stroke-dasharray: 89, 200;
        stroke-dashoffset: 0;
        stroke: dt('progressspinner.colorOne');
        animation:
            p-progressspinner-dash 1.5s ease-in-out infinite,
            p-progressspinner-color 6s ease-in-out infinite;
        stroke-linecap: round;
    }

    @keyframes p-progressspinner-rotate {
        100% {
            transform: rotate(360deg);
        }
    }
    @keyframes p-progressspinner-dash {
        0% {
            stroke-dasharray: 1, 200;
            stroke-dashoffset: 0;
        }
        50% {
            stroke-dasharray: 89, 200;
            stroke-dashoffset: -35px;
        }
        100% {
            stroke-dasharray: 89, 200;
            stroke-dashoffset: -124px;
        }
    }
    @keyframes p-progressspinner-color {
        100%,
        0% {
            stroke: dt('progressspinner.color.one');
        }
        40% {
            stroke: dt('progressspinner.color.two');
        }
        66% {
            stroke: dt('progressspinner.color.three');
        }
        80%,
        90% {
            stroke: dt('progressspinner.color.four');
        }
    }
`,cn={root:"p-progressspinner",spin:"p-progressspinner-spin",circle:"p-progressspinner-circle"},mn=b.extend({name:"progressspinner",style:dn,classes:cn}),gn={name:"BaseProgressSpinner",extends:y,props:{strokeWidth:{type:String,default:"2"},fill:{type:String,default:"none"},animationDuration:{type:String,default:"2s"}},style:mn,provide:function(){return{$pcProgressSpinner:this,$parentInstance:this}}},fn={name:"ProgressSpinner",extends:gn,inheritAttrs:!1,computed:{svgStyle:function(){return{"animation-duration":this.animationDuration}}}},hn=["fill","stroke-width"];function vn(n,a,r,s,e,t){return o(),i("div",m({class:n.cx("root"),role:"progressbar"},n.ptmi("root")),[(o(),i("svg",m({class:n.cx("spin"),viewBox:"25 25 50 50",style:t.svgStyle},n.ptm("spin")),[f("circle",m({class:n.cx("circle"),cx:"50",cy:"50",r:"20",fill:n.fill,"stroke-width":n.strokeWidth,strokeMiterlimit:"10"},n.ptm("circle")),null,16,hn)],16))],16)}fn.render=vn;export{kn as _,pn as a,fn as b,ln as c,Z as s};
//# sourceMappingURL=index-QjiiTS5s.js.map
