import{$ as m,a0 as b,ag as p,aq as v,an as c,bZ as y,b_ as w,c as d,a as u,e as D,k as h,O as l}from"./index-CUXw-u_C.js";var S=`
    .p-slider {
        display: block;
        position: relative;
        background: dt('slider.track.background');
        border-radius: dt('slider.track.border.radius');
    }

    .p-slider-handle {
        cursor: grab;
        touch-action: none;
        user-select: none;
        display: flex;
        justify-content: center;
        align-items: center;
        height: dt('slider.handle.height');
        width: dt('slider.handle.width');
        background: dt('slider.handle.background');
        border-radius: dt('slider.handle.border.radius');
        transition:
            background dt('slider.transition.duration'),
            color dt('slider.transition.duration'),
            border-color dt('slider.transition.duration'),
            box-shadow dt('slider.transition.duration'),
            outline-color dt('slider.transition.duration');
        outline-color: transparent;
    }

    .p-slider-handle::before {
        content: '';
        width: dt('slider.handle.content.width');
        height: dt('slider.handle.content.height');
        display: block;
        background: dt('slider.handle.content.background');
        border-radius: dt('slider.handle.content.border.radius');
        box-shadow: dt('slider.handle.content.shadow');
        transition: background dt('slider.transition.duration');
    }

    .p-slider:not(.p-disabled) .p-slider-handle:hover {
        background: dt('slider.handle.hover.background');
    }

    .p-slider:not(.p-disabled) .p-slider-handle:hover::before {
        background: dt('slider.handle.content.hover.background');
    }

    .p-slider-handle:focus-visible {
        box-shadow: dt('slider.handle.focus.ring.shadow');
        outline: dt('slider.handle.focus.ring.width') dt('slider.handle.focus.ring.style') dt('slider.handle.focus.ring.color');
        outline-offset: dt('slider.handle.focus.ring.offset');
    }

    .p-slider-range {
        display: block;
        background: dt('slider.range.background');
        border-radius: dt('slider.track.border.radius');
    }

    .p-slider.p-slider-horizontal {
        height: dt('slider.track.size');
    }

    .p-slider-horizontal .p-slider-range {
        inset-block-start: 0;
        inset-inline-start: 0;
        height: 100%;
    }

    .p-slider-horizontal .p-slider-handle {
        inset-block-start: 50%;
        margin-block-start: calc(-1 * calc(dt('slider.handle.height') / 2));
        margin-inline-start: calc(-1 * calc(dt('slider.handle.width') / 2));
    }

    .p-slider-vertical {
        min-height: 100px;
        width: dt('slider.track.size');
    }

    .p-slider-vertical .p-slider-handle {
        inset-inline-start: 50%;
        margin-inline-start: calc(-1 * calc(dt('slider.handle.width') / 2));
        margin-block-end: calc(-1 * calc(dt('slider.handle.height') / 2));
    }

    .p-slider-vertical .p-slider-range {
        inset-block-end: 0;
        inset-inline-start: 0;
        width: 100%;
    }
`,k={handle:{position:"absolute"},range:{position:"absolute"}},P={root:function(e){var t=e.instance,a=e.props;return["p-slider p-component",{"p-disabled":a.disabled,"p-invalid":t.$invalid,"p-slider-horizontal":a.orientation==="horizontal","p-slider-vertical":a.orientation==="vertical"}]},range:"p-slider-range",handle:"p-slider-handle"},E=m.extend({name:"slider",style:S,classes:P,inlineStyles:k}),L={name:"BaseSlider",extends:b,props:{min:{type:Number,default:0},max:{type:Number,default:100},orientation:{type:String,default:"horizontal"},step:{type:Number,default:null},range:{type:Boolean,default:!1},tabindex:{type:Number,default:0},ariaLabelledby:{type:String,default:null},ariaLabel:{type:String,default:null}},style:E,provide:function(){return{$pcSlider:this,$parentInstance:this}}};function o(n){"@babel/helpers - typeof";return o=typeof Symbol=="function"&&typeof Symbol.iterator=="symbol"?function(e){return typeof e}:function(e){return e&&typeof Symbol=="function"&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e},o(n)}function B(n,e,t){return(e=M(e))in n?Object.defineProperty(n,e,{value:t,enumerable:!0,configurable:!0,writable:!0}):n[e]=t,n}function M(n){var e=A(n,"string");return o(e)=="symbol"?e:e+""}function A(n,e){if(o(n)!="object"||!n)return n;var t=n[Symbol.toPrimitive];if(t!==void 0){var a=t.call(n,e);if(o(a)!="object")return a;throw new TypeError("@@toPrimitive must return a primitive value.")}return(e==="string"?String:Number)(n)}function V(n){return T(n)||H(n)||x(n)||z()}function z(){throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function x(n,e){if(n){if(typeof n=="string")return g(n,e);var t={}.toString.call(n).slice(8,-1);return t==="Object"&&n.constructor&&(t=n.constructor.name),t==="Map"||t==="Set"?Array.from(n):t==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t)?g(n,e):void 0}}function H(n){if(typeof Symbol<"u"&&n[Symbol.iterator]!=null||n["@@iterator"]!=null)return Array.from(n)}function T(n){if(Array.isArray(n))return g(n)}function g(n,e){(e==null||e>n.length)&&(e=n.length);for(var t=0,a=Array(e);t<e;t++)a[t]=n[t];return a}var I={name:"Slider",extends:L,inheritAttrs:!1,emits:["change","slideend"],dragging:!1,handleIndex:null,initX:null,initY:null,barWidth:null,barHeight:null,dragListener:null,dragEndListener:null,beforeUnmount:function(){this.unbindDragListeners()},methods:{updateDomData:function(){var e=this.$el.getBoundingClientRect();this.initX=e.left+y(),this.initY=e.top+w(),this.barWidth=this.$el.offsetWidth,this.barHeight=this.$el.offsetHeight},setValue:function(e){var t,a=e.touches?e.touches[0].pageX:e.pageX,s=e.touches?e.touches[0].pageY:e.pageY;this.orientation==="horizontal"?c(this.$el)?t=(this.initX+this.barWidth-a)*100/this.barWidth:t=(a-this.initX)*100/this.barWidth:t=(this.initY+this.barHeight-s)*100/this.barHeight;var i=(this.max-this.min)*(t/100)+this.min;if(this.step){var r=this.range?this.value[this.handleIndex]:this.value,f=i-r;f<0?i=r+Math.ceil(i/this.step-r/this.step)*this.step:f>0&&(i=r+Math.floor(i/this.step-r/this.step)*this.step)}else i=Math.floor(i);this.updateModel(e,i)},updateModel:function(e,t){var a=Math.round(t*100)/100,s;this.range?(s=this.value?V(this.value):[],this.handleIndex==0?(a<this.min?a=this.min:a>=this.max&&(a=this.max),s[0]=a):(a>this.max?a=this.max:a<=this.min&&(a=this.min),s[1]=a)):(a<this.min?a=this.min:a>this.max&&(a=this.max),s=a),this.writeValue(s,e),this.$emit("change",s)},onDragStart:function(e,t){this.disabled||(this.$el.setAttribute("data-p-sliding",!0),this.dragging=!0,this.updateDomData(),this.range&&this.value[0]===this.max?this.handleIndex=0:this.handleIndex=t,e.currentTarget.focus())},onDrag:function(e){this.dragging&&this.setValue(e)},onDragEnd:function(e){this.dragging&&(this.dragging=!1,this.$el.setAttribute("data-p-sliding",!1),this.$emit("slideend",{originalEvent:e,value:this.value}))},onBarClick:function(e){this.disabled||v(e.target,"data-pc-section")!=="handle"&&(this.updateDomData(),this.setValue(e))},onMouseDown:function(e,t){this.bindDragListeners(),this.onDragStart(e,t)},onKeyDown:function(e,t){switch(this.handleIndex=t,e.code){case"ArrowDown":case"ArrowLeft":this.decrementValue(e,t),e.preventDefault();break;case"ArrowUp":case"ArrowRight":this.incrementValue(e,t),e.preventDefault();break;case"PageDown":this.decrementValue(e,t,!0),e.preventDefault();break;case"PageUp":this.incrementValue(e,t,!0),e.preventDefault();break;case"Home":this.updateModel(e,this.min),e.preventDefault();break;case"End":this.updateModel(e,this.max),e.preventDefault();break}},onBlur:function(e,t){var a,s;(a=(s=this.formField).onBlur)===null||a===void 0||a.call(s,e)},decrementValue:function(e,t){var a=arguments.length>2&&arguments[2]!==void 0?arguments[2]:!1,s;this.range?this.step?s=this.value[t]-this.step:s=this.value[t]-1:this.step?s=this.value-this.step:!this.step&&a?s=this.value-10:s=this.value-1,this.updateModel(e,s),e.preventDefault()},incrementValue:function(e,t){var a=arguments.length>2&&arguments[2]!==void 0?arguments[2]:!1,s;this.range?this.step?s=this.value[t]+this.step:s=this.value[t]+1:this.step?s=this.value+this.step:!this.step&&a?s=this.value+10:s=this.value+1,this.updateModel(e,s),e.preventDefault()},bindDragListeners:function(){this.dragListener||(this.dragListener=this.onDrag.bind(this),document.addEventListener("mousemove",this.dragListener)),this.dragEndListener||(this.dragEndListener=this.onDragEnd.bind(this),document.addEventListener("mouseup",this.dragEndListener))},unbindDragListeners:function(){this.dragListener&&(document.removeEventListener("mousemove",this.dragListener),this.dragListener=null),this.dragEndListener&&(document.removeEventListener("mouseup",this.dragEndListener),this.dragEndListener=null)},rangeStyle:function(){if(this.range){var e=this.rangeEndPosition>this.rangeStartPosition?this.rangeEndPosition-this.rangeStartPosition:this.rangeStartPosition-this.rangeEndPosition,t=this.rangeEndPosition>this.rangeStartPosition?this.rangeStartPosition:this.rangeEndPosition;return this.horizontal?{"inset-inline-start":t+"%",width:e+"%"}:{bottom:t+"%",height:e+"%"}}else return this.horizontal?{width:this.handlePosition+"%"}:{height:this.handlePosition+"%"}},handleStyle:function(){return this.horizontal?{"inset-inline-start":this.handlePosition+"%"}:{bottom:this.handlePosition+"%"}},rangeStartHandleStyle:function(){return this.horizontal?{"inset-inline-start":this.rangeStartPosition+"%"}:{bottom:this.rangeStartPosition+"%"}},rangeEndHandleStyle:function(){return this.horizontal?{"inset-inline-start":this.rangeEndPosition+"%"}:{bottom:this.rangeEndPosition+"%"}}},computed:{value:function(){var e;if(this.range){var t,a,s,i;return[(t=(a=this.d_value)===null||a===void 0?void 0:a[0])!==null&&t!==void 0?t:this.min,(s=(i=this.d_value)===null||i===void 0?void 0:i[1])!==null&&s!==void 0?s:this.max]}return(e=this.d_value)!==null&&e!==void 0?e:this.min},horizontal:function(){return this.orientation==="horizontal"},vertical:function(){return this.orientation==="vertical"},handlePosition:function(){return this.value<this.min?0:this.value>this.max?100:(this.value-this.min)*100/(this.max-this.min)},rangeStartPosition:function(){return this.value&&this.value[0]!==void 0?this.value[0]<this.min?0:(this.value[0]-this.min)*100/(this.max-this.min):0},rangeEndPosition:function(){return this.value&&this.value.length===2&&this.value[1]!==void 0?this.value[1]>this.max?100:(this.value[1]-this.min)*100/(this.max-this.min):100},dataP:function(){return p(B({},this.orientation,this.orientation))}}},K=["data-p"],C=["data-p"],N=["tabindex","aria-valuemin","aria-valuenow","aria-valuemax","aria-labelledby","aria-label","aria-orientation","data-p"],X=["tabindex","aria-valuemin","aria-valuenow","aria-valuemax","aria-labelledby","aria-label","aria-orientation","data-p"],j=["tabindex","aria-valuemin","aria-valuenow","aria-valuemax","aria-labelledby","aria-label","aria-orientation","data-p"];function W(n,e,t,a,s,i){return u(),d("div",l({class:n.cx("root"),onClick:e[18]||(e[18]=function(){return i.onBarClick&&i.onBarClick.apply(i,arguments)})},n.ptmi("root"),{"data-p-sliding":!1,"data-p":i.dataP}),[D("span",l({class:n.cx("range"),style:[n.sx("range"),i.rangeStyle()]},n.ptm("range"),{"data-p":i.dataP}),null,16,C),n.range?h("",!0):(u(),d("span",l({key:0,class:n.cx("handle"),style:[n.sx("handle"),i.handleStyle()],onTouchstartPassive:e[0]||(e[0]=function(r){return i.onDragStart(r)}),onTouchmovePassive:e[1]||(e[1]=function(r){return i.onDrag(r)}),onTouchend:e[2]||(e[2]=function(r){return i.onDragEnd(r)}),onMousedown:e[3]||(e[3]=function(r){return i.onMouseDown(r)}),onKeydown:e[4]||(e[4]=function(r){return i.onKeyDown(r)}),onBlur:e[5]||(e[5]=function(r){return i.onBlur(r)}),tabindex:n.tabindex,role:"slider","aria-valuemin":n.min,"aria-valuenow":n.d_value,"aria-valuemax":n.max,"aria-labelledby":n.ariaLabelledby,"aria-label":n.ariaLabel,"aria-orientation":n.orientation},n.ptm("handle"),{"data-p":i.dataP}),null,16,N)),n.range?(u(),d("span",l({key:1,class:n.cx("handle"),style:[n.sx("handle"),i.rangeStartHandleStyle()],onTouchstartPassive:e[6]||(e[6]=function(r){return i.onDragStart(r,0)}),onTouchmovePassive:e[7]||(e[7]=function(r){return i.onDrag(r)}),onTouchend:e[8]||(e[8]=function(r){return i.onDragEnd(r)}),onMousedown:e[9]||(e[9]=function(r){return i.onMouseDown(r,0)}),onKeydown:e[10]||(e[10]=function(r){return i.onKeyDown(r,0)}),onBlur:e[11]||(e[11]=function(r){return i.onBlur(r,0)}),tabindex:n.tabindex,role:"slider","aria-valuemin":n.min,"aria-valuenow":n.d_value?n.d_value[0]:null,"aria-valuemax":n.max,"aria-labelledby":n.ariaLabelledby,"aria-label":n.ariaLabel,"aria-orientation":n.orientation},n.ptm("startHandler"),{"data-p":i.dataP}),null,16,X)):h("",!0),n.range?(u(),d("span",l({key:2,class:n.cx("handle"),style:[n.sx("handle"),i.rangeEndHandleStyle()],onTouchstartPassive:e[12]||(e[12]=function(r){return i.onDragStart(r,1)}),onTouchmovePassive:e[13]||(e[13]=function(r){return i.onDrag(r)}),onTouchend:e[14]||(e[14]=function(r){return i.onDragEnd(r)}),onMousedown:e[15]||(e[15]=function(r){return i.onMouseDown(r,1)}),onKeydown:e[16]||(e[16]=function(r){return i.onKeyDown(r,1)}),onBlur:e[17]||(e[17]=function(r){return i.onBlur(r,1)}),tabindex:n.tabindex,role:"slider","aria-valuemin":n.min,"aria-valuenow":n.d_value?n.d_value[1]:null,"aria-valuemax":n.max,"aria-labelledby":n.ariaLabelledby,"aria-label":n.ariaLabel,"aria-orientation":n.orientation},n.ptm("endHandler"),{"data-p":i.dataP}),null,16,j)):h("",!0)],16,K)}I.render=W;export{I as s};
//# sourceMappingURL=index-BrgKCV-4.js.map
