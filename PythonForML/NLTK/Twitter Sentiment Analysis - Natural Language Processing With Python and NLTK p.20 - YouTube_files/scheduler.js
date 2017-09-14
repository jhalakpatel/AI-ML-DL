(function(){var f="function"==typeof Object.create?Object.create:function(a){function b(){}
b.prototype=a;return new b},g;
if("function"==typeof Object.setPrototypeOf)g=Object.setPrototypeOf;else{var h;a:{var k={L:!0},l={};try{l.__proto__=k;h=l.L;break a}catch(a){}h=!1}g=h?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var m=g,n=this;
function p(a){a=a.split(".");for(var b=n,c=0;c<a.length;c++)if(b=b[a[c]],null==b)return null;return b}
function q(a,b,c){return a.call.apply(a.bind,arguments)}
function r(a,b,c){if(!a)throw Error();if(2<arguments.length){var d=Array.prototype.slice.call(arguments,2);return function(){var c=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(c,d);return a.apply(b,c)}}return function(){return a.apply(b,arguments)}}
function u(a,b,c){Function.prototype.bind&&-1!=Function.prototype.bind.toString().indexOf("native code")?u=q:u=r;return u.apply(null,arguments)}
var v=Date.now||function(){return+new Date};
function w(a,b){var c=a.split("."),d=n;c[0]in d||!d.execScript||d.execScript("var "+c[0]);for(var e;c.length&&(e=c.shift());)c.length||void 0===b?d[e]&&d[e]!==Object.prototype[e]?d=d[e]:d=d[e]={}:d[e]=b}
;function x(){this.h=this.h;this.m=this.m}
x.prototype.h=!1;x.prototype.dispose=function(){this.h||(this.h=!0,this.o())};
x.prototype.o=function(){if(this.m)for(;this.m.length;)this.m.shift()()};var z=window.yt&&window.yt.config_||window.ytcfg&&window.ytcfg.data_||{};w("yt.config_",z);function A(){return"SCHEDULER_SOFT_STATE_TIMER"in z?z.SCHEDULER_SOFT_STATE_TIMER:800}
;var B=1E3/60-3;function C(a){a=void 0===a?{}:a;x.call(this);this.a=[];this.a[4]=[];this.a[3]=[];this.a[2]=[];this.a[1]=[];this.a[0]=[];this.f=0;this.G=a.timeout||1;this.c={};this.l=B;this.s=this.b=this.j=0;this.u=this.i=!1;this.g=[];this.A=u(this.I,this);this.F=u(this.K,this);this.C=u(this.H,this);this.D=u(this.J,this);this.v=!1;this.B=!!window.requestIdleCallback;(this.w=!!a.useRaf&&!!window.requestAnimationFrame)&&document.addEventListener("visibilitychange",this.A)}
C.prototype=f(x.prototype);C.prototype.constructor=C;if(m)m(C,x);else for(var D in x)if("prototype"!=D)if(Object.defineProperties){var E=Object.getOwnPropertyDescriptor(x,D);E&&Object.defineProperty(C,D,E)}else C[D]=x[D];C.a=x.prototype;function F(a,b){var c=v();G(b);c=v()-c;a.i||(a.l-=c)}
function H(a,b,c,d){++a.s;if(10==c)return F(a,b),a.s;var e=a.s;a.c[e]=b;a.i&&!d?a.g.push({id:e,M:c}):(a.a[c].push(e),a.u||a.i||(0!=a.b&&I(a)!=a.j&&J(a),a.start()));return e}
function K(a){a.g.length=0;for(var b=4;0<=b;b--)a.a[b].length=0;a.c={};J(a)}
function I(a){for(var b=4;b>=a.f;b--)if(0<a.a[b].length)return 0<b?!document.hidden&&a.w?3:2:1;return 0}
function G(a){try{a()}catch(b){(a=p("yt.logging.errors.log"))&&a(b)}}
C.prototype.H=function(a){var b=void 0;a&&(b=a.timeRemaining());this.v=!0;L(this,b);this.v=!1};
C.prototype.K=function(){L(this)};
C.prototype.J=function(){L(this)};
C.prototype.I=function(){this.b&&(J(this),this.start())};
function L(a,b){J(a);a.i=!0;for(var c=v()+(b||a.l),d=a.a[4];d.length;){var e=d.shift(),t=a.c[e];delete a.c[e];t&&G(t)}d=a.v?0:1;d=a.f>d?a.f:d;if(!(v()>=c)){do{a:{e=a;t=d;for(var y=3;y>=t;y--)for(var M=e.a[y];M.length;){var N=M.shift(),O=e.c[N];delete e.c[N];if(O){e=O;break a}}e=null}e&&G(e)}while(e&&v()<c)}a.i=!1;c=0;for(d=a.g.length;c<d;c++)e=a.g[c],a.a[e.M].push(e.id);a.l=B;a:{for(c=3;0<=c;c--)if(a.a[c].length){c=!0;break a}c=!1}(c||a.g.length)&&a.start();a.g.length=0}
C.prototype.start=function(){this.u=!1;if(0==this.b)switch(this.j=I(this),this.j){case 1:var a=this.C;this.b=this.B?window.requestIdleCallback(a,{timeout:3E3}):window.setTimeout(a,300);break;case 2:this.b=window.setTimeout(this.F,this.G);break;case 3:this.b=window.requestAnimationFrame(this.D)}};
function J(a){if(a.b){switch(a.j){case 1:var b=a.b;a.B?window.cancelIdleCallback(b):window.clearTimeout(b);break;case 2:window.clearTimeout(a.b);break;case 3:window.cancelAnimationFrame(a.b)}a.b=0}}
C.prototype.o=function(){K(this);J(this);this.w&&document.removeEventListener("visibilitychange",this.A);x.prototype.o.call(this)};var P=p("yt.scheduler.instance.timerIdMap_")||{},Q=0,R=0;function S(){var a=p("ytglobal.schedulerInstanceInstance_");if(!a||a.h)a=new C(("scheduler"in z?z.scheduler:void 0)||{}),w("ytglobal.schedulerInstanceInstance_",a);return a}
function T(){var a=p("ytglobal.schedulerInstanceInstance_");a&&(a&&"function"==typeof a.dispose&&a.dispose(),w("ytglobal.schedulerInstanceInstance_",null))}
function U(){K(S())}
function V(a,b,c){if(0==c||void 0===c)return c=void 0===c,-H(S(),a,b,c);var d=window.setTimeout(function(){var c=H(S(),a,b);P[d]=c},c);
return d}
function aa(a){F(S(),a)}
function ba(a){var b=S();if(0>a)delete b.c[-a];else{var c=P[a];c?(delete b.c[c],delete P[a]):window.clearTimeout(a)}}
function W(a){var b=p("ytcsi.tick");b&&b(a)}
function ca(){W("jse");X()}
function X(){window.clearTimeout(Q);S().start()}
function da(){var a=S();J(a);a.u=!0;window.clearTimeout(Q);Q=window.setTimeout(ca,A())}
function Y(){window.clearTimeout(R);R=window.setTimeout(function(){W("jset");Z(0)},A())}
function Z(a){Y();var b=S();b.f=a;b.start()}
function ea(a){Y();var b=S();b.f>a&&(b.f=a,b.start())}
function fa(){window.clearTimeout(R);var a=S();a.f=0;a.start()}
;p("yt.scheduler.initialized")||(w("yt.scheduler.instance.dispose",T),w("yt.scheduler.instance.addJob",V),w("yt.scheduler.instance.addImmediateJob",aa),w("yt.scheduler.instance.cancelJob",ba),w("yt.scheduler.instance.cancelAllJobs",U),w("yt.scheduler.instance.start",X),w("yt.scheduler.instance.pause",da),w("yt.scheduler.instance.setPriorityThreshold",Z),w("yt.scheduler.instance.enablePriorityThreshold",ea),w("yt.scheduler.instance.clearPriorityThreshold",fa),w("yt.scheduler.initialized",!0));}).call(this);
