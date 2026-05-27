'use client';

import React, { useState } from 'react';
import { Upload, FileText, CheckCircle2, AlertCircle, Loader2, Sparkles, Diamond, Cpu, ShieldCheck, Zap, HardDrive, BarChart3, ArrowUpRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

export default function IngestHangar() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState(0);

  const stages = [
    { label: 'Verificación Biométrica', icon: ShieldCheck },
    { label: 'Parsing de Conocimiento', icon: FileText },
    { label: 'Clasificación de Nicho', icon: Cpu },
    { label: 'Indexación Neuronal', icon: Zap },
  ];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setSuccess(null);
      setStage(0);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    
    // Simular progreso de etapas
    const interval = setInterval(() => {
      setStage(prev => (prev < 3 ? prev + 1 : prev));
    }, 1500);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post('http://localhost:8000/api/ingest', formData);
      clearInterval(interval);
      setStage(3);
      setTimeout(() => {
        setSuccess(`¡Éxito! El libro "${file.name}" ha sido clasificado en el nicho: ${res.data.category}`);
        setFile(null);
      }, 500);
    } catch (err) {
      clearInterval(interval);
      console.error(err);
      setError('Error al procesar el archivo. Revisa el formato.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-16 pb-32">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-8">
        <div className="flex items-center gap-6">
          <div className="p-4 bg-purple-600 rounded-2xl shadow-[0_0_40px_rgba(147,51,234,0.4)]">
            <Upload className="w-10 h-10 text-white" />
          </div>
          <div>
            <h2 className="text-6xl font-black text-white tracking-tighter uppercase leading-none">Hangar de <span className="text-purple-400">Ingesta</span></h2>
            <p className="text-slate-400 text-lg font-medium mt-2">Alimenta el núcleo con nueva sabiduría.</p>
          </div>
        </div>

        <div className="flex gap-4">
          <div className="glass-panel px-6 py-3 rounded-2xl border-purple-500/20 bg-purple-500/5">
            <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest mb-1">Capacidad Hangar</p>
            <div className="flex items-center gap-3">
              <div className="h-1.5 w-24 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-purple-500 w-[65%]" />
              </div>
              <span className="text-xs font-bold text-white">65%</span>
            </div>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
        {/* ZONA DE CARGA PRINCIPAL */}
        <div className="lg:col-span-8 space-y-12">
          <AnimatePresence mode="wait">
            {success ? (
              <motion.div 
                key="success"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="glass-panel p-20 rounded-[3rem] border-green-500/30 bg-green-500/5 text-center relative overflow-hidden"
              >
                <div className="absolute top-0 left-0 w-full h-1 bg-green-500 shadow-[0_0_15px_rgba(34,197,94,0.5)]"></div>
                <div className="w-24 h-24 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-8">
                  <CheckCircle2 className="w-12 h-12 text-green-400" />
                </div>
                <h3 className="text-4xl font-black text-white mb-4 uppercase">Asset Asegurado</h3>
                <p className="text-green-400 text-xl font-medium mb-12 max-w-xl mx-auto">{success}</p>
                <button 
                  onClick={() => setSuccess(null)}
                  className="bg-white hover:bg-green-400 text-black font-black px-12 py-5 rounded-2xl transition-all active:scale-95 flex items-center gap-2 mx-auto"
                >
                  INGRESAR OTRO RECURSO
                  <ArrowUpRight className="w-5 h-5" />
                </button>
              </motion.div>
            ) : (
              <motion.div
                key="form"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-12"
              >
                {/* PLATAFORMA DE ESCANEO (Dropzone) */}
                <div className="relative group">
                  <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 to-cyan-500 rounded-[3rem] blur opacity-20 group-hover:opacity-40 transition duration-1000"></div>
                  <div className="relative flex flex-col items-center justify-center border-2 border-dashed border-white/10 rounded-[3rem] p-24 bg-slate-900/50 hover:bg-slate-900 transition-all cursor-pointer">
                    <input
                      type="file"
                      onChange={handleFileChange}
                      className="absolute inset-0 opacity-0 cursor-pointer z-10"
                    />
                    
                    <div className="relative mb-8">
                      <div className="absolute inset-0 bg-purple-500/20 blur-3xl rounded-full animate-pulse"></div>
                      <div className="relative p-8 bg-white/5 rounded-[2rem] border border-white/10 group-hover:scale-110 group-hover:border-purple-500/50 transition-all duration-500">
                        {file ? <FileText className="w-20 h-20 text-purple-400" /> : <Upload className="w-20 h-20 text-slate-700" />}
                      </div>
                    </div>

                    <div className="text-center">
                      <h3 className="text-3xl font-black text-white mb-2 uppercase tracking-tight">
                        {file ? file.name : 'Plataforma de Escaneo'}
                      </h3>
                      <p className="text-slate-500 text-lg font-medium">
                        {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB • Listo para procesar` : 'Arrastra el conocimiento aquí o haz click'}
                      </p>
                    </div>

                    {/* Corner accents */}
                    <div className="absolute top-6 left-6 w-8 h-8 border-t-2 border-l-2 border-white/10 rounded-tl-xl"></div>
                    <div className="absolute top-6 right-6 w-8 h-8 border-t-2 border-r-2 border-white/10 rounded-tr-xl"></div>
                    <div className="absolute bottom-6 left-6 w-8 h-8 border-b-2 border-l-2 border-white/10 rounded-bl-xl"></div>
                    <div className="absolute bottom-6 right-6 w-8 h-8 border-b-2 border-r-2 border-white/10 rounded-br-xl"></div>
                  </div>
                </div>

                {/* PIPELINE DE PROCESAMIENTO */}
                <div className="glass-panel p-10 rounded-[2.5rem] border-white/5 bg-white/[0.01]">
                  <div className="flex justify-between items-center mb-8">
                    <h4 className="text-xs font-black text-slate-500 uppercase tracking-[0.3em]">Pipeline de Ingesta</h4>
                    {loading && (
                      <span className="text-[10px] font-bold text-purple-400 animate-pulse uppercase tracking-widest">IA en Proceso...</span>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-4 gap-4">
                    {stages.map((s, i) => (
                      <div key={i} className={`flex flex-col items-center gap-3 transition-opacity duration-500 ${loading && i > stage ? 'opacity-20' : 'opacity-100'}`}>
                        <div className={`p-4 rounded-2xl border ${loading && i === stage ? 'border-purple-500 bg-purple-500/10 animate-pulse' : (loading && i < stage) || success ? 'border-green-500/50 bg-green-500/10' : 'border-white/5 bg-white/5'} transition-all`}>
                          <s.icon className={`w-6 h-6 ${loading && i === stage ? 'text-purple-400' : (loading && i < stage) || success ? 'text-green-400' : 'text-slate-600'}`} />
                        </div>
                        <span className={`text-[9px] font-black uppercase text-center tracking-tighter ${loading && i === stage ? 'text-white' : 'text-slate-600'}`}>{s.label}</span>
                      </div>
                    ))}
                  </div>

                  <div className="mt-10 h-2 bg-slate-800 rounded-full overflow-hidden p-0.5">
                    <motion.div 
                      className="h-full bg-gradient-to-r from-purple-600 to-cyan-500 rounded-full"
                      initial={{ width: '0%' }}
                      animate={{ width: loading ? `${(stage + 1) * 25}%` : success ? '100%' : '0%' }}
                    />
                  </div>
                </div>

                <button
                  onClick={handleUpload}
                  disabled={!file || loading}
                  className="w-full bg-white hover:bg-purple-500 text-black hover:text-white font-black py-8 rounded-[2rem] text-2xl shadow-2xl transition-all disabled:opacity-20 active:scale-[0.98] flex items-center justify-center gap-4 group"
                >
                  {loading ? (
                    <Loader2 className="w-8 h-8 animate-spin" />
                  ) : (
                    <Sparkles className="w-8 h-8 group-hover:scale-125 transition-transform" />
                  )}
                  {loading ? 'ALIMENTANDO EL NÚCLEO...' : 'INICIAR INGESTA DE DATOS'}
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* SIDEBAR DE DIAGNÓSTICOS */}
        <div className="lg:col-span-4 space-y-8">
          <div className="glass-panel p-8 rounded-[2rem] border-white/5">
            <div className="flex items-center gap-3 mb-6">
              <HardDrive className="w-5 h-5 text-cyan-400" />
              <h4 className="font-black text-white text-sm uppercase tracking-widest">Almacenamiento</h4>
            </div>
            <div className="space-y-6">
              <div>
                <div className="flex justify-between text-xs mb-2">
                  <span className="text-slate-500">Espacio Usado</span>
                  <span className="text-white font-bold">1.2 TB / 2.0 TB</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-500 w-[60%]" />
                </div>
              </div>
              <div className="p-4 rounded-xl bg-cyan-500/5 border border-cyan-500/10">
                <p className="text-[10px] text-cyan-400 font-bold uppercase tracking-widest mb-1">Optimización</p>
                <p className="text-xs text-slate-300 leading-relaxed">Deduplicación activa. Ahorro de un 15% en almacenamiento neuronal.</p>
              </div>
            </div>
          </div>

          <div className="glass-panel p-8 rounded-[2rem] border-white/5">
            <div className="flex items-center gap-3 mb-6">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              <h4 className="font-black text-white text-sm uppercase tracking-widest">Actividad</h4>
            </div>
            <div className="space-y-4">
              {[
                { label: 'Libros procesados hoy', val: '14' },
                { label: 'Tiempo prom. de ingesta', val: '4.2s' },
                { label: 'Precisión de nicho', val: '98.5%' }
              ].map((item, i) => (
                <div key={i} className="flex justify-between items-center py-2 border-b border-white/5 last:border-0">
                  <span className="text-xs text-slate-500">{item.label}</span>
                  <span className="text-xs font-bold text-white font-mono">{item.val}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {error && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-10 right-10 flex items-center gap-4 bg-red-500 text-white px-8 py-4 rounded-2xl shadow-2xl z-50 font-bold"
        >
          <AlertCircle className="w-6 h-6" />
          {error}
        </motion.div>
      )}
    </div>
  );
}
