import{$ as S,af as k,a0 as O,ag as w,aK as m,ao as z,W as A,c as p,a as g,e as h,a2 as y,n as V,k as I,O as s,z as B,ac as f,aQ as c,N as C,F as K,m as $,l as x,aS as D,h as E}from"./index-CUXw-u_C.js";var R=`
    .p-togglebutton {
        display: inline-flex;
        cursor: pointer;
        user-select: none;
        overflow: hidden;
        position: relative;
        color: dt('togglebutton.color');
        background: dt('togglebutton.background');
        border: 1px solid dt('togglebutton.border.color');
        padding: dt('togglebutton.padding');
        font-size: 1rem;
        font-family: inherit;
        font-feature-settings: inherit;
        transition:
            background dt('togglebutton.transition.duration'),
            color dt('togglebutton.transition.duration'),
            border-color dt('togglebutton.transition.duration'),
            outline-color dt('togglebutton.transition.duration'),
            box-shadow dt('togglebutton.transition.duration');
        border-radius: dt('togglebutton.border.radius');
        outline-color: transparent;
        font-weight: dt('togglebutton.font.weight');
    }

    .p-togglebutton-content {
        display: inline-flex;
        flex: 1 1 auto;
        align-items: center;
        justify-content: center;
        gap: dt('togglebutton.gap');
        padding: dt('togglebutton.content.padding');
        background: transparent;
        border-radius: dt('togglebutton.content.border.radius');
        transition:
            background dt('togglebutton.transition.duration'),
            color dt('togglebutton.transition.duration'),
            border-color dt('togglebutton.transition.duration'),
            outline-color dt('togglebutton.transition.duration'),
            box-shadow dt('togglebutton.transition.duration');
    }

    .p-togglebutton:not(:disabled):not(.p-togglebutton-checked):hover {
        background: dt('togglebutton.hover.background');
        color: dt('togglebutton.hover.color');
    }

    .p-togglebutton.p-togglebutton-checked {
        background: dt('togglebutton.checked.background');
        border-color: dt('togglebutton.checked.border.color');
        color: dt('togglebutton.checked.color');
    }

    .p-togglebutton-checked .p-togglebutton-content {
        background: dt('togglebutton.content.checked.background');
        box-shadow: dt('togglebutton.content.checked.shadow');
    }

    .p-togglebutton:focus-visible {
        box-shadow: dt('togglebutton.focus.ring.shadow');
        outline: dt('togglebutton.focus.ring.width') dt('togglebutton.focus.ring.style') dt('togglebutton.focus.ring.color');
        outline-offset: dt('togglebutton.focus.ring.offset');
    }

    .p-togglebutton.p-invalid {
        border-color: dt('togglebutton.invalid.border.color');
    }

    .p-togglebutton:disabled {
        opacity: 1;
        cursor: default;
        background: dt('togglebutton.disabled.background');
        border-color: dt('togglebutton.disabled.border.color');
        color: dt('togglebutton.disabled.color');
    }

    .p-togglebutton-label,
    .p-togglebutton-icon {
        position: relative;
        transition: none;
    }

    .p-togglebutton-icon {
        color: dt('togglebutton.icon.color');
    }

    .p-togglebutton:not(:disabled):not(.p-togglebutton-checked):hover .p-togglebutton-icon {
        color: dt('togglebutton.icon.hover.color');
    }

    .p-togglebutton.p-togglebutton-checked .p-togglebutton-icon {
        color: dt('togglebutton.icon.checked.color');
    }

    .p-togglebutton:disabled .p-togglebutton-icon {
        color: dt('togglebutton.icon.disabled.color');
    }

    .p-togglebutton-sm {
        padding: dt('togglebutton.sm.padding');
        font-size: dt('togglebutton.sm.font.size');
    }

    .p-togglebutton-sm .p-togglebutton-content {
        padding: dt('togglebutton.content.sm.padding');
    }

    .p-togglebutton-lg {
        padding: dt('togglebutton.lg.padding');
        font-size: dt('togglebutton.lg.font.size');
    }

    .p-togglebutton-lg .p-togglebutton-content {
        padding: dt('togglebutton.content.lg.padding');
    }

    .p-togglebutton-fluid {
        width: 100%;
    }
`,j={root:function(e){var n=e.instance,l=e.props;return["p-togglebutton p-component",{"p-togglebutton-checked":n.active,"p-invalid":n.$invalid,"p-togglebutton-fluid":l.fluid,"p-togglebutton-sm p-inputfield-sm":l.size==="small","p-togglebutton-lg p-inputfield-lg":l.size==="large"}]},content:"p-togglebutton-content",icon:"p-togglebutton-icon",label:"p-togglebutton-label"},F=S.extend({name:"togglebutton",style:R,classes:j}),N={name:"BaseToggleButton",extends:O,props:{onIcon:String,offIcon:String,onLabel:{type:String,default:"Yes"},offLabel:{type:String,default:"No"},readonly:{type:Boolean,default:!1},tabindex:{type:Number,default:null},ariaLabelledby:{type:String,default:null},ariaLabel:{type:String,default:null},size:{type:String,default:null},fluid:{type:Boolean,default:null}},style:F,provide:function(){return{$pcToggleButton:this,$parentInstance:this}}};function b(t){"@babel/helpers - typeof";return b=typeof Symbol=="function"&&typeof Symbol.iterator=="symbol"?function(e){return typeof e}:function(e){return e&&typeof Symbol=="function"&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e},b(t)}function q(t,e,n){return(e=_(e))in t?Object.defineProperty(t,e,{value:n,enumerable:!0,configurable:!0,writable:!0}):t[e]=n,t}function _(t){var e=H(t,"string");return b(e)=="symbol"?e:e+""}function H(t,e){if(b(t)!="object"||!t)return t;var n=t[Symbol.toPrimitive];if(n!==void 0){var l=n.call(t,e);if(b(l)!="object")return l;throw new TypeError("@@toPrimitive must return a primitive value.")}return(e==="string"?String:Number)(t)}var L={name:"ToggleButton",extends:N,inheritAttrs:!1,emits:["change"],methods:{getPTOptions:function(e){var n=e==="root"?this.ptmi:this.ptm;return n(e,{context:{active:this.active,disabled:this.disabled}})},onChange:function(e){!this.disabled&&!this.readonly&&(this.writeValue(!this.d_value,e),this.$emit("change",e))},onBlur:function(e){var n,l;(n=(l=this.formField).onBlur)===null||n===void 0||n.call(l,e)}},computed:{active:function(){return this.d_value===!0},hasLabel:function(){return m(this.onLabel)&&m(this.offLabel)},label:function(){return this.hasLabel?this.d_value?this.onLabel:this.offLabel:" "},dataP:function(){return w(q({checked:this.active,invalid:this.$invalid},this.size,this.size))}},directives:{ripple:k}},W=["tabindex","disabled","aria-pressed","aria-label","aria-labelledby","data-p-checked","data-p-disabled","data-p"],M=["data-p"];function Q(t,e,n,l,a,o){var i=z("ripple");return A((g(),p("button",s({type:"button",class:t.cx("root"),tabindex:t.tabindex,disabled:t.disabled,"aria-pressed":t.d_value,onClick:e[0]||(e[0]=function(){return o.onChange&&o.onChange.apply(o,arguments)}),onBlur:e[1]||(e[1]=function(){return o.onBlur&&o.onBlur.apply(o,arguments)})},o.getPTOptions("root"),{"aria-label":t.ariaLabel,"aria-labelledby":t.ariaLabelledby,"data-p-checked":o.active,"data-p-disabled":t.disabled,"data-p":o.dataP}),[h("span",s({class:t.cx("content")},o.getPTOptions("content"),{"data-p":o.dataP}),[y(t.$slots,"default",{},function(){return[y(t.$slots,"icon",{value:t.d_value,class:V(t.cx("icon"))},function(){return[t.onIcon||t.offIcon?(g(),p("span",s({key:0,class:[t.cx("icon"),t.d_value?t.onIcon:t.offIcon]},o.getPTOptions("icon")),null,16)):I("",!0)]}),h("span",s({class:t.cx("label")},o.getPTOptions("label")),B(o.label),17)]})],16,M)],16,W)),[[i]])}L.render=Q;var U=`
    .p-selectbutton {
        display: inline-flex;
        user-select: none;
        vertical-align: bottom;
        outline-color: transparent;
        border-radius: dt('selectbutton.border.radius');
    }

    .p-selectbutton .p-togglebutton {
        border-radius: 0;
        border-width: 1px 1px 1px 0;
    }

    .p-selectbutton .p-togglebutton:focus-visible {
        position: relative;
        z-index: 1;
    }

    .p-selectbutton .p-togglebutton:first-child {
        border-inline-start-width: 1px;
        border-start-start-radius: dt('selectbutton.border.radius');
        border-end-start-radius: dt('selectbutton.border.radius');
    }

    .p-selectbutton .p-togglebutton:last-child {
        border-start-end-radius: dt('selectbutton.border.radius');
        border-end-end-radius: dt('selectbutton.border.radius');
    }

    .p-selectbutton.p-invalid {
        outline: 1px solid dt('selectbutton.invalid.border.color');
        outline-offset: 0;
    }

    .p-selectbutton-fluid {
        width: 100%;
    }
    
    .p-selectbutton-fluid .p-togglebutton {
        flex: 1 1 0;
    }
`,Y={root:function(e){var n=e.props,l=e.instance;return["p-selectbutton p-component",{"p-invalid":l.$invalid,"p-selectbutton-fluid":n.fluid}]}},G=S.extend({name:"selectbutton",style:U,classes:Y}),J={name:"BaseSelectButton",extends:O,props:{options:Array,optionLabel:null,optionValue:null,optionDisabled:null,multiple:Boolean,allowEmpty:{type:Boolean,default:!0},dataKey:null,ariaLabelledby:{type:String,default:null},size:{type:String,default:null},fluid:{type:Boolean,default:null}},style:G,provide:function(){return{$pcSelectButton:this,$parentInstance:this}}};function X(t,e){var n=typeof Symbol<"u"&&t[Symbol.iterator]||t["@@iterator"];if(!n){if(Array.isArray(t)||(n=T(t))||e){n&&(t=n);var l=0,a=function(){};return{s:a,n:function(){return l>=t.length?{done:!0}:{done:!1,value:t[l++]}},e:function(d){throw d},f:a}}throw new TypeError(`Invalid attempt to iterate non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}var o,i=!0,r=!1;return{s:function(){n=n.call(t)},n:function(){var d=n.next();return i=d.done,d},e:function(d){r=!0,o=d},f:function(){try{i||n.return==null||n.return()}finally{if(r)throw o}}}}function Z(t){return nt(t)||et(t)||T(t)||tt()}function tt(){throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function T(t,e){if(t){if(typeof t=="string")return v(t,e);var n={}.toString.call(t).slice(8,-1);return n==="Object"&&t.constructor&&(n=t.constructor.name),n==="Map"||n==="Set"?Array.from(t):n==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?v(t,e):void 0}}function et(t){if(typeof Symbol<"u"&&t[Symbol.iterator]!=null||t["@@iterator"]!=null)return Array.from(t)}function nt(t){if(Array.isArray(t))return v(t)}function v(t,e){(e==null||e>t.length)&&(e=t.length);for(var n=0,l=Array(e);n<e;n++)l[n]=t[n];return l}var ot={name:"SelectButton",extends:J,inheritAttrs:!1,emits:["change"],methods:{getOptionLabel:function(e){return this.optionLabel?c(e,this.optionLabel):e},getOptionValue:function(e){return this.optionValue?c(e,this.optionValue):e},getOptionRenderKey:function(e){return this.dataKey?c(e,this.dataKey):this.getOptionLabel(e)},isOptionDisabled:function(e){return this.optionDisabled?c(e,this.optionDisabled):!1},isOptionReadonly:function(e){if(this.allowEmpty)return!1;var n=this.isSelected(e);return this.multiple?n&&this.d_value.length===1:n},onOptionSelect:function(e,n,l){var a=this;if(!(this.disabled||this.isOptionDisabled(n)||this.isOptionReadonly(n))){var o=this.isSelected(n),i=this.getOptionValue(n),r;if(this.multiple)if(o){if(r=this.d_value.filter(function(u){return!f(u,i,a.equalityKey)}),!this.allowEmpty&&r.length===0)return}else r=this.d_value?[].concat(Z(this.d_value),[i]):[i];else{if(o&&!this.allowEmpty)return;r=o?null:i}this.writeValue(r,e),this.$emit("change",{originalEvent:e,value:r})}},isSelected:function(e){var n=!1,l=this.getOptionValue(e);if(this.multiple){if(this.d_value){var a=X(this.d_value),o;try{for(a.s();!(o=a.n()).done;){var i=o.value;if(f(i,l,this.equalityKey)){n=!0;break}}}catch(r){a.e(r)}finally{a.f()}}}else n=f(this.d_value,l,this.equalityKey);return n}},computed:{equalityKey:function(){return this.optionValue?null:this.dataKey},dataP:function(){return w({invalid:this.$invalid})}},directives:{ripple:k},components:{ToggleButton:L}},lt=["aria-labelledby","data-p"];function rt(t,e,n,l,a,o){var i=C("ToggleButton");return g(),p("div",s({class:t.cx("root"),role:"group","aria-labelledby":t.ariaLabelledby},t.ptmi("root"),{"data-p":o.dataP}),[(g(!0),p(K,null,$(t.options,function(r,u){return g(),x(i,{key:o.getOptionRenderKey(r),modelValue:o.isSelected(r),onLabel:o.getOptionLabel(r),offLabel:o.getOptionLabel(r),disabled:t.disabled||o.isOptionDisabled(r),unstyled:t.unstyled,size:t.size,readonly:o.isOptionReadonly(r),onChange:function(P){return o.onOptionSelect(P,r,u)},pt:t.ptm("pcToggleButton")},D({_:2},[t.$slots.option?{name:"default",fn:E(function(){return[y(t.$slots,"option",{option:r,index:u},function(){return[h("span",s({ref_for:!0},t.ptm("pcToggleButton").label),B(o.getOptionLabel(r)),17)]})]}),key:"0"}:void 0]),1032,["modelValue","onLabel","offLabel","disabled","unstyled","size","readonly","onChange","pt"])}),128))],16,lt)}ot.render=rt;export{ot as s};
//# sourceMappingURL=index-B4PwScSb.js.map
