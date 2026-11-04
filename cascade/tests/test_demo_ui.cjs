// Run with: node --test cascade/tests/test_demo_ui.cjs
// Exercise result transitions without a browser or model downloads.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../demo.html'),'utf8');

class Element {
  constructor(){this.children=[];this.dataset={};this.style={};this.attributes={};this.hidden=false;this.currentTime=0;this.files=[];this.className='';this._text='';this.classList={toggle:()=>{}};}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join(' ');}
  append(...children){for(const child of children){child.parent=this;this.children.push(child);}}
  replaceChildren(...children){this.children=[];this._text='';this.append(...children);}
  setAttribute(key,value){this.attributes[key]=value;}
  remove(){this.parent.children=this.parent.children.filter(c=>c!==this);}
  pause(){}
  play(){return Promise.resolve();}
}
function page(){
  const elements=new Map([...html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)].map(m=>{const el=new Element();el.hidden=/\bhidden\b/.test(m[0]);return [m[1],el];}));
  elements.get('analysis-mode').value='offline';
  const document={createElement:()=>new Element(),getElementById:id=>{assert.ok(elements.has(id),'Missing DOM ID: '+id);return elements.get(id);}};
  const ctx=vm.createContext({document,fetch:()=>new Promise(()=>{}),setTimeout:()=>1,clearTimeout:()=>{},URL,URLSearchParams,console});
  vm.runInContext(html.split('<script>')[1].split('</script>')[0],ctx);
  return {ctx,el:id=>elements.get(id),run:code=>vm.runInContext(code,ctx)};
}
const trace={stage:1,detector:'xception',score:0.91,low:0.2,high:0.8,decision:'FAKE'};
const windowJob={id:'stream-test',phase:'complete',report:{windows:[
  {start_seconds:0,end_seconds:4,outcome:'decided',decision:'FAKE',score:0.91,trace:[trace]},
  {start_seconds:4,end_seconds:8,outcome:'skipped',reason:'過載',trace:[]}
]}};

test('Early Exit exposes the actual decision and leaves later models unexecuted',()=>{
  const p=page();p.ctx.trace=[trace];p.run('renderCascade(trace,null,true)');
  const rows=p.el('cascade-path').children;
  assert.match(rows[0].textContent,/Xception.*0\.9100.*疑似偽造/);
  assert.match(rows[1].textContent,/UCF.*未執行/);
  assert.match(rows[2].textContent,/SBI.*未執行/);
  assert.doesNotMatch(rows[1].textContent,/Fake score/);
});

test('skipped windows never inherit the previous REAL or FAKE result',()=>{
  const p=page();p.ctx.job=windowJob;p.run('renderTimeline(job)');
  assert.match(p.el('window-verdict').textContent,/FAKE/);
  p.el('timeline').children[1].onclick();
  assert.equal(p.el('window-verdict').textContent,'未分析／過載');
  assert.doesNotMatch(p.el('cascade-path').textContent,/0\.9100/);
  assert.equal(p.el('player').currentTime,4);
  assert.equal(p.el('timeline').children[1].attributes['aria-pressed'],'true');
});

test('active streaming selection preserves playback and button focus across updates',()=>{
  const p=page();p.ctx.job=structuredClone(windowJob);p.ctx.job.phase='playing';p.run('renderTimeline(job)');
  const button=p.el('timeline').children[1];p.el('player').currentTime=2;
  button.onclick();p.run('renderTimeline(job)');
  assert.equal(p.el('player').currentTime,2);
  assert.equal(p.el('timeline').children[1],button);
  assert.equal(button.attributes['aria-pressed'],'true');
  assert.equal(p.el('window-verdict').textContent,'未分析／過載');
});

test('starting another analysis clears previous verdict, evidence and window selection',()=>{
  const p=page();p.ctx.job=windowJob;p.run('renderTimeline(job)');
  p.el('results').hidden=false;p.el('evidence').hidden=false;p.run('clearResult()');
  assert.equal(p.el('results').hidden,true);
  assert.equal(p.el('evidence').hidden,true);
  assert.equal(p.el('stream-focus').hidden,true);
  assert.equal(p.el('decision-empty').hidden,false);
  assert.equal(p.el('timeline').children.length,0);
  assert.doesNotMatch(p.el('cascade-path').textContent,/0\.9100/);
});

test('incomplete final windows retain their sub-second duration in the timeline',()=>{
  const p=page();p.ctx.job={...windowJob,report:{windows:[{...windowJob.report.windows[0],start_seconds:12,end_seconds:12.03}]}};
  p.run('renderTimeline(job)');
  assert.match(p.el('timeline').children[0].textContent,/12–12\.03 s/);
});

test('changing analysis mode clears the previous mode result',()=>{
  const p=page();p.ctx.job=windowJob;p.run('renderTimeline(job)');
  p.el('analysis-mode').value='offline';p.el('analysis-mode').onchange({type:'change'});
  assert.equal(p.el('stream-focus').hidden,true);
  assert.equal(p.el('decision-empty').hidden,false);
  assert.equal(p.el('mode-label').textContent,'離線分析');
});

test('streaming renders the selected policy subset and exposes its identity',()=>{
  const p=page();p.ctx.job=structuredClone(windowJob);
  p.ctx.job.report.policy={policy_id:'ucf-only',stages:[{detector:'ucf',low:.5,high:.5}]};
  p.ctx.job.report.windows[0].trace=[{...trace,detector:'ucf'}];
  p.run('showStream(job)');
  assert.equal(p.el('cascade-path').children.length,1);
  assert.match(p.el('cascade-path').textContent,/UCF/);
  assert.doesNotMatch(p.el('cascade-path').textContent,/Xception|SBI/);
  assert.match(p.el('stream-summary').textContent,/ucf-only/);
});
