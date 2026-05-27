'use client';

import React, { useState, useEffect } from 'react';
import { ShieldAlert, Activity, Database, AlertTriangle, CheckCircle2, Search, Filter, ArrowUpRight, Lock, Eye, Trash2, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

export default function AuditCenter() {
  const [loading, setLoading] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [isScanning, setIsScanning] = useState(false);

  const stats = [
    { label: 'Integridad de Datos', value: '99.8%', icon: Database, color: 'text-cyan-400' },
    { label: 'Archivos Huérfanos', value: '12', icon: AlertTriangle, color: 'text-yellow-400' },
    { label: 'Duplicados Detectados', value: '3', icon: ShieldAlert, color: 'text-red-400' },
  ];

  const handleScan = () => {
    setIsScanning(true);
    setScanProgress(0);
    const interval = setInterval(() => {
      setScanProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsScanning(false);
          return 100;
        }
        return prev + 2;
      });
    }, 50);
  };

  return (
    <div className="space-y-16 pb-32">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-8">
        <div className="flex items-center gap-6">
          <div className="p-4 bg-cyan-600 rounded-2xl shadow-[0_0_40px_rgba(6,182,212,0.4)]">
            <Activity className="w-10 h-10 text-white" />
          </div>
          <div>
            <h2 className="text-6xl font-black text-white tracking-tighter uppercase leading-none">Centro de <span className="text-cyan-400">Auditoría</span></h2>
            <p className="text-slate-400 text-lg font-medium mt-2">Monitoreo de integridad y salud de la bóveda.</p>
          </div>
        </div>

        <button 
          onClick={handleScan}
          disabled={isScanning}
          className="glass-panel px-8 py-4 rounded-2xl border-cyan-500/20 bg-cyan-500/5 hover:bg-cyan-500/10 transition-all flex items-center gap-3 group"
        >
          <RefreshCw className={`w-5 h-5 text-cyan-400 ${isScanning ? 'animate-spin' : 'group-hover:rotate-180 transition-transform duration-500'}`} />
          <span className="text-sm font-black text-white uppercase tracking-widest">Ejecutar Escaneo Total</span>
        </button>
      </header>

      {/* METRICAS DE SALUD */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {stats.map((stat, i) => (
          <div key={i} className="glass-panel p-8 rounded-[2.5rem] border-white/5 bg-white/[0.02] relative overflow-hidden group">
            <div className={`absolute top-0 right-0 p-6 opacity-5 group-hover:opacity-10 transition-opacity`}>
              <stat.icon className="w-24 h-24" />
            </div>
            <div className="relative z-10">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-6 bg-slate-800/50 ${stat.color}`}>
                <stat.icon className="w-6 h-6" />
              </div>
              <p className="text-xs text-slate-500 font-black uppercase tracking-widest mb-1">{stat.label}</p>
              <p className="text-4xl font-black text-white">{stat.value}</p>
            </div>
          </div>
        ))}
      </div>

      {isScanning && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-10 rounded-[2.5rem] border-cyan-500/20 bg-cyan-500/5"
        >
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-cyan-500 animate-ping"></div>
              <h4 className="text-sm font-black text-white uppercase tracking-widest">Escaneo de Integridad en Progreso...</h4>
            </div>
            <span className="text-2xl font-black text-cyan-400 font-mono">{scanProgress}%</span>
          </div>
          <div className="h-4 bg-slate-900 rounded-full overflow-hidden p-1 border border-white/5">
            <motion.div 
              className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 rounded-full"
              initial={{ width: '0%' }}
              animate={{ width: `${scanProgress}%` }}
            />
          </div>
          <div className="mt-6 grid grid-cols-4 gap-4">
            {['Checksums', 'Deduplicación', 'Metadata', 'Vectores'].map((task, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className={`w-1.5 h-1.5 rounded-full ${scanProgress > (i + 1) * 25 ? 'bg-green-500' : 'bg-slate-700'}`}></div>
                <span className="text-[10px] font-bold text-slate-500 uppercase">{task}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* REPORTE DE INCIDENCIAS */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
        <div className="lg:col-span-8 space-y-8">
          <div className="glass-panel rounded-[2.5rem] border-white/5 overflow-hidden">
            <div className="p-8 border-b border-white/5 flex justify-between items-center bg-white/[0.01]">
              <h3 className="text-xl font-black text-white uppercase tracking-tight">Incidencias Críticas</h3>
              <div className="flex gap-2">
                <span className="px-3 py-1 rounded-full bg-red-500/10 text-red-400 text-[10px] font-black uppercase tracking-widest border border-red-500/20">3 Prioridad Alta</span>
              </div>
            </div>
            <div className="divide-y divide-white/5">
              {[
                { file: 'Estrategias_Venta_2024.pdf', issue: 'Duplicado detectado en Nicho: Ventas', icon: ShieldAlert, color: 'text-red-400' },
                { file: 'Metodos_Cialdini_Full.txt', issue: 'Falta indexación vectorial en 3 páginas', icon: AlertTriangle, color: 'text-yellow-400' },
                { file: 'FB_Ads_Masterclass.mp4', issue: 'Metadata corrupta o incompleta', icon: Lock, color: 'text-purple-400' },
              ].map((item, i) => (
                <div key={i} className="p-8 hover:bg-white/[0.02] transition-colors flex items-center justify-between group">
                  <div className="flex items-center gap-6">
                    <div className={`p-4 rounded-2xl bg-slate-800/50 ${item.color}`}>
                      <item.icon className="w-6 h-6" />
                    </div>
                    <div>
                      <h4 className="text-white font-bold group-hover:text-cyan-400 transition-colors">{item.file}</h4>
                      <p className="text-slate-500 text-sm">{item.issue}</p>
                    </div>
                  </div>
                  <div className="flex gap-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="p-3 rounded-xl bg-white/5 hover:bg-cyan-500/20 text-slate-400 hover:text-cyan-400 transition-all">
                      <Eye className="w-5 h-5" />
                    </button>
                    <button className="p-3 rounded-xl bg-white/5 hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-all">
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <div className="p-8 bg-slate-900/50 text-center">
              <button className="text-xs font-black text-cyan-400 uppercase tracking-widest hover:text-white transition-colors">Ver todos los reportes de seguridad</button>
            </div>
          </div>
        </div>

        <div className="lg:col-span-4 space-y-8">
          <div className="glass-panel p-8 rounded-[2.5rem] border-white/5 bg-gradient-to-br from-cyan-500/10 to-transparent">
            <Lock className="w-10 h-10 text-cyan-400 mb-6" />
            <h3 className="text-2xl font-black text-white uppercase mb-2 tracking-tight">Auto-Sanación</h3>
            <p className="text-slate-400 text-sm leading-relaxed mb-8">
              El sistema ha corregido automáticamente <span className="text-white font-bold">14 errores</span> de segmentación en las últimas 24 horas.
            </p>
            <div className="p-4 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center gap-4">
              <CheckCircle2 className="w-5 h-5 text-cyan-400" />
              <span className="text-xs font-bold text-white">Protocolo v2.4 Activo</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
