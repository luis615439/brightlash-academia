import React, { useState } from 'react';

const OfferValidator = () => {
  const [status, setStatus] = useState('IDLE');
  
  const handleValidation = () => {
    setStatus('VALIDATING');
    setTimeout(() => setStatus('GREEN'), 1500);
  };

  return (
    <div className="min-h-screen bg-[#0a0f18] text-slate-200 p-8 flex flex-col lg:flex-row gap-8 font-sans">
      {/* Sidebar Panel - Agent Levy */}
      <aside className="w-full lg:w-1/3 bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-[0_8px_32px_rgba(0,0,0,0.6)] p-6 flex flex-col gap-4">
        <div className="flex items-center gap-3 border-b border-slate-700/50 pb-4">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 shadow-[0_0_15px_rgba(99,102,241,0.5)] flex items-center justify-center font-bold text-lg">
            LV
          </div>
          <div>
            <h2 className="text-xl font-semibold tracking-tight text-white">Agent Levy</h2>
            <p className="text-xs text-indigo-400 font-medium">ONBOARDING AI</p>
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto space-y-4 pt-4">
          <div className="bg-slate-800/50 rounded-lg p-3 text-sm border border-slate-700/30">
            Awaiting offer input for structural validation. Target: High-Ticket SaaS.
          </div>
        </div>
      </aside>

      {/* Main Validation Area */}
      <main className="w-full lg:w-2/3 bg-slate-900/40 backdrop-blur-md border border-slate-700/40 rounded-2xl shadow-[0_8px_32px_rgba(0,0,0,0.4)] flex flex-col">
        <header className="px-6 py-5 border-b border-slate-800/60 flex justify-between items-center">
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-200 to-slate-400">
            Offer Validation Engine
          </h1>
          <span className={`px-3 py-1 text-xs font-bold rounded-full ${
            status === 'GREEN' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 
            status === 'VALIDATING' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30 animate-pulse' : 
            'bg-slate-800 text-slate-400 border border-slate-700'
          }`}>
            {status}
          </span>
        </header>

        <section className="p-6 flex-1 flex flex-col gap-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Target Niche</label>
              <input type="text" defaultValue="B2B SaaS" className="w-full bg-slate-950/50 border border-slate-700/50 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-indigo-500/50 transition-colors" />
            </div>
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Price Point</label>
              <input type="text" defaultValue="$2,500/mo" className="w-full bg-slate-950/50 border border-slate-700/50 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-indigo-500/50 transition-colors" />
            </div>
          </div>
          
          <div className="space-y-2">
            <label className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Core Value Proposition</label>
            <textarea rows="4" defaultValue="Implementamos infraestructura de IA autónoma que reduce los costos operativos en un 40%." className="w-full bg-slate-950/50 border border-slate-700/50 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-indigo-500/50 transition-colors resize-none"></textarea>
          </div>

          <div className="mt-auto pt-6 border-t border-slate-800/60">
            <button 
              onClick={handleValidation}
              className="w-full relative overflow-hidden group bg-slate-800 border border-slate-600/50 hover:border-indigo-500/50 text-white rounded-lg px-4 py-3 font-semibold tracking-wide transition-all duration-300 shadow-[0_0_20px_rgba(0,0,0,0.3)] hover:shadow-[0_0_25px_rgba(99,102,241,0.2)]"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-600/20 to-purple-600/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
              <span className="relative">INITIATE VALIDATION SEQUENCE</span>
            </button>
          </div>
        </section>
      </main>
    </div>
  );
};

export default OfferValidator;
