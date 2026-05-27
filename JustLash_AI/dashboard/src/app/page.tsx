'use client';

import React, { useState, useEffect } from 'react';
import Sidebar from '@/components/Sidebar';
import NicheExplorer from '@/components/NicheExplorer';
import IdeaLab from '@/components/IdeaLab';
import IngestHangar from '@/components/IngestHangar';
import AuditCenter from '@/components/AuditCenter';
import { motion, AnimatePresence } from 'framer-motion';
import { Database, Book, ShieldAlert, Zap, TrendingUp } from 'lucide-react';
import axios from 'axios';

function TopStats() {
  const [stats, setStats] = useState({ total_files: 0, total_chunks: 0, categories: [] });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await axios.get('http://localhost:8000/api/stats');
        setStats(res.data);
      } catch (err) {
        console.error(err);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
      {[
        { label: 'Bóveda Total', value: stats.total_files, icon: Book, color: 'text-cyan-400', bg: 'bg-cyan-500/5', border: 'border-cyan-500/20' },
        { label: 'Libros Indexados', value: stats.total_chunks, icon: Database, color: 'text-purple-400', bg: 'bg-purple-500/5', border: 'border-purple-500/20' },
        { label: 'Nichos Activos', value: stats.categories.length, icon: Zap, color: 'text-yellow-400', bg: 'bg-yellow-500/5', border: 'border-yellow-500/20' },
      ].map((stat, i) => (
        <div key={i} className={`glass-panel p-10 rounded-[2.5rem] flex items-center gap-8 border ${stat.border} ${stat.bg} group hover:scale-[1.02] transition-all duration-500`}>
          <div className={`p-6 rounded-2xl bg-slate-900 ${stat.color} group-hover:scale-110 transition-transform shadow-2xl`}>
            <stat.icon className="w-10 h-10" />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <p className="text-[10px] text-slate-500 font-black uppercase tracking-[0.3em]">{stat.label}</p>
              <TrendingUp className="w-3 h-3 text-green-500 opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
            <p className="text-6xl font-black text-white tabular-nums tracking-tighter">{stat.value}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState('niche');

  return (
    <main className="flex h-screen w-screen bg-[#020617] overflow-hidden">
      {/* Sidebar - Columna Izquierda Rígida */}
      <aside className="w-[380px] h-full flex-shrink-0">
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      </aside>
      
      {/* Contenido - Columna Derecha con Scroll */}
      <section className="flex-1 h-full overflow-y-auto relative">
        {/* Animated Background Grid */}
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-fixed bg-center opacity-20 pointer-events-none"></div>
        <div className="absolute inset-0 bg-gradient-to-b from-[#020617] via-transparent to-[#020617] pointer-events-none"></div>

        <div className="p-20 max-w-[1500px] mx-auto relative z-10">
          <TopStats />
          
          <div className="mt-24">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              >
                {activeTab === 'niche' && <NicheExplorer />}
                {activeTab === 'search' && <IdeaLab />}
                {activeTab === 'ingest' && <IngestHangar />}
                {activeTab === 'audit' && <AuditCenter />}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </section>
    </main>
  );
}
