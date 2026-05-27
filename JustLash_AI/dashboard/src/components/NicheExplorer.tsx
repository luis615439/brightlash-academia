'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Folder, FileText, ChevronRight, ArrowLeft, Search, Database, Layers, Sparkles, Download, Eye, Zap, Crown, Target, Heart, Brain, X, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const nicheIcons: Record<string, any> = {
  'FINANZAS_Y_RIQUEZA': Crown,
  'MARKETING_Y_VENTAS': Target,
  'IA_Y_AUTOMATIZACION': Zap,
  'PSICOLOGIA_Y_PNL': Brain,
  'ESPIRITUALIDAD_Y_BIENESTAR': Heart,
  'DEFAULT': Folder
};

interface Category {
  name: string;
  count: number;
}

function PreviewModal({ filename, onClose }: { filename: string, onClose: () => void }) {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchContent = async () => {
      try {
        const res = await axios.get(`http://localhost:8000/api/files/content?filename=${encodeURIComponent(filename)}`);
        if (res.data.content && res.data.content.trim().length > 0) {
          setContent(res.data.content);
        } else {
          setContent('⚠️ El archivo parece estar dañado o vacío (contiene solo bytes nulos). No se pudo extraer texto.');
        }
      } catch (err) {
        setContent('⚠️ No se pudo cargar el contenido del archivo. Asegúrate de que el backend esté corriendo y el archivo exista.');
      } finally {
        setLoading(false);
      }
    };
    fetchContent();
  }, [filename]);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-[100] flex items-center justify-center p-10 bg-black/80 backdrop-blur-sm"
    >
      <motion.div 
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        className="bg-slate-900 border border-white/10 w-full max-w-5xl h-[80vh] rounded-[3rem] overflow-hidden flex flex-col shadow-[0_0_100px_rgba(0,0,0,0.5)]"
      >
        <div className="p-8 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-cyan-500/10 rounded-xl">
              <FileText className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-xl font-black text-white uppercase tracking-tight">{filename}</h3>
              <p className="text-[10px] text-slate-500 font-bold uppercase tracking-[0.2em]">Vista Previa de Activo Digital</p>
            </div>
          </div>
          <button onClick={onClose} className="p-4 hover:bg-white/10 rounded-full transition-colors">
            <X className="w-6 h-6 text-slate-400" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-12">
          {loading ? (
            <div className="h-full flex flex-col items-center justify-center gap-6">
              <Loader2 className="w-12 h-12 text-cyan-500 animate-spin" />
              <p className="text-slate-500 font-mono text-sm animate-pulse">Escaneando contenido binario...</p>
            </div>
          ) : (
            <div className="bg-slate-950/50 p-10 rounded-[2rem] border border-white/5">
               <pre className="whitespace-pre-wrap text-slate-300 font-medium text-lg leading-relaxed font-sans">
                 {content}
               </pre>
               <div className="mt-10 pt-10 border-t border-white/5 text-center">
                  <p className="text-[10px] text-slate-600 font-black uppercase tracking-widest italic">Fin del extracto de vista previa</p>
               </div>
            </div>
          )}
        </div>

        <div className="p-8 border-t border-white/5 bg-white/[0.02] flex justify-end gap-4">
           <button 
             onClick={() => window.open(`http://localhost:8000/api/files/download?filename=${encodeURIComponent(filename)}`, '_blank')}
             className="px-10 py-4 bg-cyan-500 text-black font-black rounded-2xl flex items-center gap-3 hover:bg-cyan-400 transition-all shadow-xl"
            >
             <Download className="w-5 h-5" /> DESCARGAR COMPLETO
           </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

export default function NicheExplorer() {
  const [categories, setCategories] = useState<Category[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [files, setFiles] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [previewFile, setPreviewFile] = useState<string | null>(null);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/stats');
      setCategories(res.data.categories);
      setLoading(false);
    } catch (err) {
      console.error("Error fetching stats:", err);
      setLoading(false);
    }
  };

  const fetchFiles = async (category: string) => {
    setLoading(true);
    setSelectedCategory(category);
    try {
      const res = await axios.get(`http://localhost:8000/api/files?category=${category}`);
      setFiles(res.data);
      setLoading(false);
    } catch (err) {
      console.error("Error fetching files:", err);
      setLoading(false);
    }
  };

  const handleDownload = (filename: string) => {
    window.open(`http://localhost:8000/api/files/download?filename=${encodeURIComponent(filename)}`, '_blank');
  };

  const filteredFiles = files.filter(f => 
    f.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-16 pb-32">
      <AnimatePresence>
        {previewFile && <PreviewModal filename={previewFile} onClose={() => setPreviewFile(null)} />}
      </AnimatePresence>

      <header className="flex flex-col md:flex-row md:items-end justify-between gap-8">
        <div className="flex items-center gap-6">
          <div className="p-4 bg-cyan-600 rounded-2xl shadow-[0_0_40px_rgba(6,182,212,0.4)]">
            <Layers className="w-10 h-10 text-white" />
          </div>
          <div>
            <h2 className="text-6xl font-black text-white tracking-tighter uppercase leading-none">Explorador de <span className="text-cyan-400">Nichos</span></h2>
            <p className="text-slate-400 text-lg font-medium mt-2">
              {selectedCategory ? `Explorando activos en ${selectedCategory.replace(/_/g, ' ')}` : 'Navega la arquitectura de conocimiento de la bóveda.'}
            </p>
          </div>
        </div>

        {selectedCategory && (
          <button 
            onClick={() => setSelectedCategory(null)}
            className="flex items-center gap-3 px-8 py-4 rounded-2xl bg-white/5 border border-white/10 text-white font-black hover:bg-white/10 transition-all active:scale-95 shadow-xl"
          >
            <ArrowLeft className="w-5 h-5 text-cyan-400" />
            VOLVER A CATEGORÍAS
          </button>
        )}
      </header>
      
      <AnimatePresence mode="wait">
        {!selectedCategory ? (
          <motion.div 
            key="categories"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
          >
            {categories.map((cat, i) => {
              const Icon = nicheIcons[cat.name] || nicheIcons['DEFAULT'];
              return (
                <motion.div 
                  key={cat.name} 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="glass-panel p-10 rounded-[3rem] group relative overflow-hidden bg-slate-900/40 border-white/5 hover:border-cyan-500/40 transition-all duration-700 shadow-2xl"
                >
                  <div className="absolute -right-4 -bottom-4 opacity-5 group-hover:opacity-10 group-hover:scale-125 transition-all duration-700">
                    <Icon className="w-40 h-40 text-white" />
                  </div>
                  
                  <div className="relative z-10">
                    <div className="flex justify-between items-start mb-10">
                      <div className="p-5 bg-cyan-500/10 rounded-2xl group-hover:bg-cyan-500 text-cyan-400 group-hover:text-black transition-all shadow-xl">
                        <Icon className="w-8 h-8" />
                      </div>
                      <div className="flex flex-col items-end">
                        <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest mb-1 leading-tight">Activos</span>
                        <span className="text-2xl font-black text-white">{cat.count}</span>
                      </div>
                    </div>
                    
                    <h3 className="text-3xl font-black text-white mb-6 uppercase tracking-tight group-hover:text-cyan-400 transition-colors leading-tight">
                      {cat.name.replace(/_/g, ' ')}
                    </h3>
                    
                    <button 
                      onClick={() => fetchFiles(cat.name)}
                      className="w-full flex items-center justify-between px-8 py-5 rounded-2xl bg-white/5 group-hover:bg-cyan-500 border border-white/5 group-hover:border-cyan-500 text-slate-400 group-hover:text-black font-black uppercase tracking-widest text-xs transition-all shadow-lg"
                    >
                      ABRIR SECTOR
                      <ChevronRight className="w-5 h-5" />
                    </button>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        ) : (
          <motion.div 
            key="files"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-12"
          >
            {/* FILE EXPLORER TOOLBAR */}
            <div className="flex flex-col md:flex-row gap-6">
              <div className="relative flex-1 group">
                <div className="absolute -inset-0.5 bg-cyan-500/20 rounded-2xl blur opacity-0 group-focus-within:opacity-100 transition-opacity"></div>
                <div className="relative flex items-center bg-slate-900/90 border border-white/10 rounded-2xl shadow-2xl">
                  <Search className="ml-6 w-6 h-6 text-cyan-500/50" />
                  <input 
                    type="text" 
                    placeholder="Filtrar por nombre de archivo..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-transparent border-none py-6 px-6 text-white text-xl font-bold focus:outline-none placeholder:text-slate-700"
                  />
                </div>
              </div>
              <div className="glass-panel px-8 py-4 rounded-2xl flex items-center gap-4 border-purple-500/20 bg-purple-500/5">
                <Database className="w-6 h-6 text-purple-400" />
                <div className="flex flex-col">
                  <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest leading-tight">Total en Lote</span>
                  <span className="text-xl font-black text-white tabular-nums">{files.length}</span>
                </div>
              </div>
            </div>

            {/* PREMIUM DATA TABLE - WITH FUNCTIONAL BUTTONS */}
            <div className="glass-panel rounded-[4rem] border-white/5 overflow-hidden bg-slate-900/40 shadow-[0_0_50px_rgba(0,0,0,0.5)]">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-white/[0.03] border-b border-white/5">
                      <th className="px-12 py-8 text-[10px] font-black text-slate-500 uppercase tracking-[0.3em]">Identificador de Recurso</th>
                      <th className="px-12 py-8 text-[10px] font-black text-slate-500 uppercase tracking-[0.3em]">Lote</th>
                      <th className="px-12 py-8 text-[10px] font-black text-slate-500 uppercase tracking-[0.3em]">Estado</th>
                      <th className="px-12 py-8 text-[10px] font-black text-slate-500 uppercase tracking-[0.3em] text-right">Acciones</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {filteredFiles.map((file, idx) => (
                      <motion.tr 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: idx * 0.02 }}
                        key={idx} 
                        className="hover:bg-cyan-500/[0.03] transition-all group cursor-pointer"
                        onClick={() => setPreviewFile(file.filename)}
                      >
                        <td className="px-12 py-10">
                          <div className="flex items-center gap-6">
                            <div className="p-4 bg-slate-800 rounded-2xl group-hover:bg-cyan-500/20 group-hover:scale-110 transition-all shadow-lg">
                              <FileText className="w-6 h-6 text-slate-600 group-hover:text-cyan-400" />
                            </div>
                            <div>
                              <p className="text-white text-lg font-black group-hover:text-cyan-400 transition-colors leading-none mb-2">{file.filename}</p>
                              <p className="text-slate-600 text-[10px] uppercase font-black tracking-widest">Digital Asset • Vault ID: {Math.random().toString(36).substring(7).toUpperCase()}</p>
                            </div>
                          </div>
                        </td>
                        <td className="px-12 py-10">
                          <span className="font-mono text-xs text-slate-400 font-bold bg-white/5 border border-white/10 px-4 py-2 rounded-xl shadow-inner">
                            {selectedCategory?.toUpperCase()}-{file.batch_id.toString().padStart(2, '0')}
                          </span>
                        </td>
                        <td className="px-12 py-10">
                          <div className="flex items-center gap-3">
                            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
                            <span className="text-[10px] font-black text-white uppercase tracking-widest">Protegido</span>
                          </div>
                        </td>
                        <td className="px-12 py-10 text-right">
                          <div className="flex items-center justify-end gap-3 opacity-0 group-hover:opacity-100 transition-all translate-x-4 group-hover:translate-x-0">
                            <button 
                              onClick={(e) => { e.stopPropagation(); setPreviewFile(file.filename); }}
                              className="px-5 py-3 rounded-xl bg-white/5 hover:bg-cyan-500 text-slate-400 hover:text-black font-black text-[10px] uppercase tracking-widest transition-all shadow-xl"
                            >
                              EXPLORAR
                            </button>
                            <button 
                              onClick={(e) => { e.stopPropagation(); handleDownload(file.filename); }}
                              className="px-5 py-3 rounded-xl bg-white/5 hover:bg-purple-500 text-slate-400 hover:text-white font-black text-[10px] uppercase tracking-widest transition-all shadow-xl"
                            >
                              BAJAR
                            </button>
                          </div>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              {filteredFiles.length === 0 && (
                <div className="p-32 text-center">
                  <Search className="w-24 h-24 text-slate-800 mx-auto mb-10 opacity-20" />
                  <p className="text-3xl font-black text-slate-700 uppercase tracking-tighter">Bóveda Vacía para este filtro</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
