import{$ as u,as as b,ag as l,ac as s,c,a as p,e,O as r}from"./index-CUXw-u_C.js";var h=`
    .p-radiobutton {
        position: relative;
        display: inline-flex;
        user-select: none;
        vertical-align: bottom;
        width: dt('radiobutton.width');
        height: dt('radiobutton.height');
    }

    .p-radiobutton-input {
        cursor: pointer;
        appearance: none;
        position: absolute;
        top: 0;
        inset-inline-start: 0;
        width: 100%;
        height: 100%;
        padding: 0;
        margin: 0;
        opacity: 0;
        z-index: 1;
        outline: 0 none;
        border: 1px solid transparent;
        border-radius: 50%;
    }

    .p-radiobutton-box {
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 50%;
        border: 1px solid dt('radiobutton.border.color');
        background: dt('radiobutton.background');
        width: dt('radiobutton.width');
        height: dt('radiobutton.height');
        transition:
            background dt('radiobutton.transition.duration'),
            color dt('radiobutton.transition.duration'),
            border-color dt('radiobutton.transition.duration'),
            box-shadow dt('radiobutton.transition.duration'),
            outline-color dt('radiobutton.transition.duration');
        outline-color: transparent;
        box-shadow: dt('radiobutton.shadow');
    }

    .p-radiobutton-icon {
        transition-duration: dt('radiobutton.transition.duration');
        background: transparent;
        font-size: dt('radiobutton.icon.size');
        width: dt('radiobutton.icon.size');
        height: dt('radiobutton.icon.size');
        border-radius: 50%;
        backface-visibility: hidden;
        transform: translateZ(0) scale(0.1);
    }

    .p-radiobutton:not(.p-disabled):has(.p-radiobutton-input:hover) .p-radiobutton-box {
        border-color: dt('radiobutton.hover.border.color');
    }

    .p-radiobutton-checked .p-radiobutton-box {
        border-color: dt('radiobutton.checked.border.color');
        background: dt('radiobutton.checked.background');
    }

    .p-radiobutton-checked .p-radiobutton-box .p-radiobutton-icon {
        background: dt('radiobutton.icon.checked.color');
        transform: translateZ(0) scale(1, 1);
        visibility: visible;
    }

    .p-radiobutton-checked:not(.p-disabled):has(.p-radiobutton-input:hover) .p-radiobutton-box {
        border-color: dt('radiobutton.checked.hover.border.color');
        background: dt('radiobutton.checked.hover.background');
    }

    .p-radiobutton:not(.p-disabled):has(.p-radiobutton-input:hover).p-radiobutton-checked .p-radiobutton-box .p-radiobutton-icon {
        background: dt('radiobutton.icon.checked.hover.color');
    }

    .p-radiobutton:not(.p-disabled):has(.p-radiobutton-input:focus-visible) .p-radiobutton-box {
        border-color: dt('radiobutton.focus.border.color');
        box-shadow: dt('radiobutton.focus.ring.shadow');
        outline: dt('radiobutton.focus.ring.width') dt('radiobutton.focus.ring.style') dt('radiobutton.focus.ring.color');
        outline-offset: dt('radiobutton.focus.ring.offset');
    }

    .p-radiobutton-checked:not(.p-disabled):has(.p-radiobutton-input:focus-visible) .p-radiobutton-box {
        border-color: dt('radiobutton.checked.focus.border.color');
    }

    .p-radiobutton.p-invalid > .p-radiobutton-box {
        border-color: dt('radiobutton.invalid.border.color');
    }

    .p-radiobutton.p-variant-filled .p-radiobutton-box {
        background: dt('radiobutton.filled.background');
    }

    .p-radiobutton.p-variant-filled.p-radiobutton-checked .p-radiobutton-box {
        background: dt('radiobutton.checked.background');
    }

    .p-radiobutton.p-variant-filled:not(.p-disabled):has(.p-radiobutton-input:hover).p-radiobutton-checked .p-radiobutton-box {
        background: dt('radiobutton.checked.hover.background');
    }

    .p-radiobutton.p-disabled {
        opacity: 1;
    }

    .p-radiobutton.p-disabled .p-radiobutton-box {
        background: dt('radiobutton.disabled.background');
        border-color: dt('radiobutton.checked.disabled.border.color');
    }

    .p-radiobutton-checked.p-disabled .p-radiobutton-box .p-radiobutton-icon {
        background: dt('radiobutton.icon.disabled.color');
    }

    .p-radiobutton-sm,
    .p-radiobutton-sm .p-radiobutton-box {
        width: dt('radiobutton.sm.width');
        height: dt('radiobutton.sm.height');
    }

    .p-radiobutton-sm .p-radiobutton-icon {
        font-size: dt('radiobutton.icon.sm.size');
        width: dt('radiobutton.icon.sm.size');
        height: dt('radiobutton.icon.sm.size');
    }

    .p-radiobutton-lg,
    .p-radiobutton-lg .p-radiobutton-box {
        width: dt('radiobutton.lg.width');
        height: dt('radiobutton.lg.height');
    }

    .p-radiobutton-lg .p-radiobutton-icon {
        font-size: dt('radiobutton.icon.lg.size');
        width: dt('radiobutton.icon.lg.size');
        height: dt('radiobutton.icon.lg.size');
    }
`,f={root:function(t){var o=t.instance,a=t.props;return["p-radiobutton p-component",{"p-radiobutton-checked":o.checked,"p-disabled":a.disabled,"p-invalid":o.$pcRadioButtonGroup?o.$pcRadioButtonGroup.$invalid:o.$invalid,"p-variant-filled":o.$variant==="filled","p-radiobutton-sm p-inputfield-sm":a.size==="small","p-radiobutton-lg p-inputfield-lg":a.size==="large"}]},box:"p-radiobutton-box",input:"p-radiobutton-input",icon:"p-radiobutton-icon"},g=u.extend({name:"radiobutton",style:h,classes:f}),v={name:"BaseRadioButton",extends:b,props:{value:null,binary:Boolean,readonly:{type:Boolean,default:!1},tabindex:{type:Number,default:null},inputId:{type:String,default:null},inputClass:{type:[String,Object],default:null},inputStyle:{type:Object,default:null},ariaLabelledby:{type:String,default:null},ariaLabel:{type:String,default:null}},style:g,provide:function(){return{$pcRadioButton:this,$parentInstance:this}}};function d(n){"@babel/helpers - typeof";return d=typeof Symbol=="function"&&typeof Symbol.iterator=="symbol"?function(t){return typeof t}:function(t){return t&&typeof Symbol=="function"&&t.constructor===Symbol&&t!==Symbol.prototype?"symbol":typeof t},d(n)}function m(n,t,o){return(t=y(t))in n?Object.defineProperty(n,t,{value:o,enumerable:!0,configurable:!0,writable:!0}):n[t]=o,n}function y(n){var t=k(n,"string");return d(t)=="symbol"?t:t+""}function k(n,t){if(d(n)!="object"||!n)return n;var o=n[Symbol.toPrimitive];if(o!==void 0){var a=o.call(n,t);if(d(a)!="object")return a;throw new TypeError("@@toPrimitive must return a primitive value.")}return(t==="string"?String:Number)(n)}var x={name:"RadioButton",extends:v,inheritAttrs:!1,emits:["change","focus","blur"],inject:{$pcRadioButtonGroup:{default:void 0}},methods:{getPTOptions:function(t){var o=t==="root"?this.ptmi:this.ptm;return o(t,{context:{checked:this.checked,disabled:this.disabled}})},onChange:function(t){if(!this.disabled&&!this.readonly){var o=this.binary?!this.checked:this.value;this.$pcRadioButtonGroup?this.$pcRadioButtonGroup.writeValue(o,t):this.writeValue(o,t),this.$emit("change",t)}},onFocus:function(t){this.$emit("focus",t)},onBlur:function(t){var o,a;this.$emit("blur",t),(o=(a=this.formField).onBlur)===null||o===void 0||o.call(a,t)}},computed:{groupName:function(){return this.$pcRadioButtonGroup?this.$pcRadioButtonGroup.groupName:this.$formName},checked:function(){var t=this.$pcRadioButtonGroup?this.$pcRadioButtonGroup.d_value:this.d_value;return t!=null&&(this.binary?!!t:s(t,this.value))},dataP:function(){return l(m({invalid:this.$invalid,checked:this.checked,disabled:this.disabled,filled:this.$variant==="filled"},this.size,this.size))}}},B=["data-p-checked","data-p-disabled","data-p"],w=["id","value","name","checked","tabindex","disabled","readonly","aria-labelledby","aria-label","aria-invalid"],P=["data-p"],z=["data-p"];function S(n,t,o,a,R,i){return p(),c("div",r({class:n.cx("root")},i.getPTOptions("root"),{"data-p-checked":i.checked,"data-p-disabled":n.disabled,"data-p":i.dataP}),[e("input",r({id:n.inputId,type:"radio",class:[n.cx("input"),n.inputClass],style:n.inputStyle,value:n.value,name:i.groupName,checked:i.checked,tabindex:n.tabindex,disabled:n.disabled,readonly:n.readonly,"aria-labelledby":n.ariaLabelledby,"aria-label":n.ariaLabel,"aria-invalid":n.invalid||void 0,onFocus:t[0]||(t[0]=function(){return i.onFocus&&i.onFocus.apply(i,arguments)}),onBlur:t[1]||(t[1]=function(){return i.onBlur&&i.onBlur.apply(i,arguments)}),onChange:t[2]||(t[2]=function(){return i.onChange&&i.onChange.apply(i,arguments)})},i.getPTOptions("input")),null,16,w),e("div",r({class:n.cx("box")},i.getPTOptions("box"),{"data-p":i.dataP}),[e("div",r({class:n.cx("icon")},i.getPTOptions("icon"),{"data-p":i.dataP}),null,16,z)],16,P)],16,B)}x.render=S;export{x as s};
//# sourceMappingURL=index-Cx3uspZw.js.map
