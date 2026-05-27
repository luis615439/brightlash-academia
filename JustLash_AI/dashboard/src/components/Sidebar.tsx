import React from 'react';
import { 
  Library, 
  Search, 
  PlusCircle, 
  ShieldCheck, 
  Settings, 
  LogOut,
  Diamond,
  ChevronRight
} from 'lucide-react';
import { motion } from 'framer-motion';

const menuItems = [
  { id: 'niche', name: 'Explorador de Nichos', icon: Library, color: 'cyan' },
  { id: 'search', name: 'Laboratorio de Ideas', icon: Search, color: 'cyan' },
  { id: 'ingest', name: 'Hangar de Ingesta', icon: PlusCircle, color: 'purple' },
  { id: 'audit', name: 'Centro de Auditoría', icon: ShieldCheck, color: 'emerald' },
];

export default function Sidebar({ activeTab, setActiveTab }: { activeTab: string, setActiveTab: (id: string) => void }) {
  return (
    <div className="flex flex-col h-screen p-10 bg-slate-950 border-r border-white/5 shadow-[40px_0_80px_rgba(0,0,0,0.9)] relative z-20 w-full">
      {/* GLOW DECORATIVO */}
      <div className="absolute top-0 left-0 w-full h-1/2 bg-gradient-to-b from-cyan-500/5 to-transparent pointer-events-none"></div>
      
      <div className="flex items-center gap-6 mb-24 relative z-10">
        <div className="p-5 bg-cyan-500 rounded-[1.5rem] shadow-[0_0_35px_rgba(6,182,212,0.6)] transform -rotate-3 hover:rotate-0 transition-transform duration-700">
          <Diamond className="w-12 h-12 text-black animate-diamond" />
        </div>
        <div className="flex flex-col">
          <h1 className="text-4xl font-black tracking-tighter text-white leading-none">
            DIAMOND
          </h1>
          <span className="text-cyan-400 font-black tracking-[0.5em] text-[11px] mt-1">VAULT CORE</span>
        </div>
      </div>

      <nav className="flex-1 space-y-4 relative z-10">
        <p className="text-[10px] font-black text-slate-600 uppercase tracking-[0.4em] ml-4 mb-8">Sistemas de Control</p>
        {menuItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={`w-full group relative flex items-center justify-between px-8 py-6 rounded-[2rem] transition-all duration-700 ${
              activeTab === item.id 
                ? 'bg-white text-black font-black shadow-[0_25px_50px_rgba(0,0,0,0.4)] scale-[1.05]' 
                : 'text-slate-500 hover:text-white hover:bg-white/5'
            }`}
          >
            <div className="flex items-center gap-6">
              <div className={`p-2 rounded-xl transition-colors ${activeTab === item.id ? 'bg-cyan-500/10' : 'bg-transparent'}`}>
                <item.icon className={`w-7 h-7 transition-colors ${activeTab === item.id ? 'text-cyan-600' : 'text-slate-600 group-hover:text-cyan-400'}`} />
              </div>
              <span className="text-sm uppercase tracking-[0.15em] font-black">{item.name}</span>
            </div>
            {activeTab === item.id && (
              <motion.div layoutId="active-pill" className="absolute left-0 w-2 h-10 bg-cyan-500 rounded-r-full shadow-[0_0_20px_rgba(6,182,212,0.9)]"></motion.div>
            )}
            <ChevronRight className={`w-5 h-5 opacity-0 group-hover:opacity-100 transition-opacity ${activeTab === item.id ? 'text-black/10' : 'text-slate-800'}`} />
          </button>
        ))}
      </nav>

      <div className="pt-10 border-t border-white/5 space-y-8 relative z-10">
        <div className="p-8 rounded-[2.5rem] bg-gradient-to-br from-slate-900 to-slate-950 border border-white/5 relative overflow-hidden group shadow-2xl">
          <div className="absolute top-0 right-0 p-6 opacity-5 group-hover:opacity-10 transition-opacity scale-150">
            <Settings className="w-16 h-16 text-white" />
          </div>
          <p className="text-[10px] text-cyan-400 font-black uppercase tracking-[0.3em] mb-4">Núcleo de Inteligencia</p>
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-4 h-4 bg-emerald-500 rounded-full animate-ping absolute opacity-40"></div>
              <div className="w-4 h-4 bg-emerald-500 rounded-full relative"></div>
            </div>
            <span className="text-sm text-white font-mono font-black tracking-widest uppercase">Sistema Activo</span>
          </div>
        </div>

        <button className="w-full flex items-center justify-center gap-4 px-8 py-5 rounded-2xl bg-white/5 hover:bg-red-500 text-slate-600 hover:text-white border border-transparent hover:border-red-500/20 transition-all font-black text-[10px] uppercase tracking-[0.3em] shadow-xl">
          <LogOut className="w-5 h-5" />
          Desconectar Sesión
        </button>
      </div>
    </div>
  );
}
