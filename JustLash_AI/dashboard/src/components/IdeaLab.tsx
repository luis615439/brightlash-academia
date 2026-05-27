'use client';

import React, { useState, useEffect } from 'react';
import { Search, Sparkles, Database, BookOpen, Zap, Target, TrendingUp, Heart, Brain, Crown, ArrowRight, ListChecks, FileText, Lightbulb, ChevronRight, Eye, Download, Share2, X, Loader2 } from 'lucide-react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const nicheStyles: Record<string, { color: string, icon: any, bg: string, border: string }> = {
  'FINANZAS_Y_RIQUEZA': { color: 'text-yellow-400', icon: Crown, bg: 'from-yellow-500/20 to-orange-600/20', border: 'border-yellow-500/30' },
  'MARKETING_Y_VENTAS': { color: 'text-cyan-400', icon: Target, bg: 'from-cyan-500/20 to-blue-600/20', border: 'border-cyan-500/30' },
  'IA_Y_AUTOMATIZACION': { color: 'text-purple-400', icon: Zap, bg: 'from-purple-500/20 to-indigo-600/20', border: 'border-purple-500/30' },
  'PSICOLOGIA_Y_PNL': { color: 'text-pink-400', icon: Brain, bg: 'from-pink-500/20 to-rose-600/20', border: 'border-pink-500/30' },
  'ESPIRITUALIDAD_Y_BIENESTAR': { color: 'text-emerald-400', icon: Heart, bg: 'from-emerald-500/20 to-teal-600/20', border: 'border-emerald-500/30' },
  'DEFAULT': { color: 'text-slate-400', icon: BookOpen, bg: 'from-slate-500/20 to-slate-700/20', border: 'border-slate-500/30' }
};

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
              <p className="text-slate-500 font-mono text-sm animate-pulse">Consultando base de vectores...</p>
            </div>
          ) : (
            <div className="bg-slate-950/50 p-10 rounded-[2rem] border border-white/5">
               <pre className="whitespace-pre-wrap text-slate-300 font-medium text-lg leading-relaxed font-sans">
                 {content}
               </pre>
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

export default function IdeaLab() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [niches, setNiches] = useState<any[]>([]);
  const [previewFile, setPreviewFile] = useState<string | null>(null);

  useEffect(() => {
    const fetchNiches = async () => {
      try {
        const res = await axios.get('http://localhost:8000/api/stats');
        setNiches(res.data.categories);
      } catch (err) {
        console.error(err);
      }
    };
    fetchNiches();
  }, []);

  const handleSearch = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const finalQuery = customQuery || query;
    if (!finalQuery.trim()) return;
    
    setLoading(true);
    setResults(null);
    try {
      const res = await axios.post(`http://localhost:8000/api/search`, {
        text: finalQuery,
        top_k: 5
      });
      setResults({
        answer: res.data.response,
        sources: res.data.sources && res.data.sources.length > 0 ? res.data.sources : []
      });
    } catch (err) {
      console.error(err);
      setResults({
        answer: '⚠️ Error al consultar la sabiduría de la bóveda. El servidor no respondió correctamente.',
        sources: []
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (filename: string) => {
    window.open(`http://localhost:8000/api/files/download?filename=${encodeURIComponent(filename)}`, '_blank');
  };

  return (
    <div className="space-y-16 pb-32">
      <AnimatePresence>
        {previewFile && <PreviewModal filename={previewFile} onClose={() => setPreviewFile(null)} />}
      </AnimatePresence>

      <header className="relative flex flex-col md:flex-row md:items-end justify-between gap-8">
        <div className="flex items-center gap-6">
          <div className="p-4 bg-cyan-500 rounded-2xl shadow-[0_0_40px_rgba(6,182,212,0.4)]">
            <Sparkles className="w-10 h-10 text-black" />
          </div>
          <div>
            <h2 className="text-6xl font-black text-white tracking-tighter uppercase leading-none">Laboratorio de <span className="text-cyan-400">Ideas</span></h2>
            <p className="text-slate-400 text-lg font-medium mt-2">Extrae sabiduría pura de la bóveda digital.</p>
          </div>
        </div>
        
        <div className="glass-panel px-6 py-3 rounded-2xl flex items-center gap-4 border-cyan-500/20 bg-cyan-500/5">
          <Database className="w-5 h-5 text-cyan-400" />
          <div className="flex flex-col">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest leading-tight">Estado de Bóveda</span>
            <span className="text-sm text-white font-mono font-bold">12,673 Libros Indexados</span>
          </div>
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,211,238,0.8)]"></div>
        </div>
      </header>

      {/* BARRA DE BÚSQUEDA PREMIUM */}
      <section className="relative">
        <form onSubmit={handleSearch} className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-[2.5rem] blur opacity-25 group-hover:opacity-50 transition duration-1000 group-focus-within:opacity-100"></div>
          <div className="relative flex items-center bg-slate-900/90 backdrop-blur-xl rounded-[2.5rem] border border-white/20 overflow-hidden shadow-2xl">
            <div className="relative flex-1 group">
              <Search className="absolute left-8 top-1/2 -translate-y-1/2 w-8 h-8 text-cyan-500/50" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="¿Qué sabiduría buscas hoy? (ej. Estrategias de persuasión en ventas)"
                className="w-full bg-slate-900/80 border border-slate-700/50 rounded-[2.5rem] py-10 pl-20 pr-10 text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 transition-all font-medium text-3xl"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="mr-4 bg-white hover:bg-cyan-400 text-black font-black px-12 py-6 rounded-[1.8rem] text-xl transition-all active:scale-95 disabled:opacity-50 flex items-center gap-3 group/btn shadow-[0_0_30px_rgba(34,211,238,0.2)]"
            >
              {loading ? (
                <div className="w-6 h-6 border-4 border-black/20 border-t-black rounded-full animate-spin"></div>
              ) : (
                <>
                  EXTRAER SABIDURÍA
                  <Sparkles className="w-6 h-6 group-hover/btn:rotate-12 transition-transform" />
                </>
              )}
            </button>
          </div>
        </form>
      </section>

      {/* GRILLA DE NICHOS REIMAGINADA */}
      <section className="space-y-8">
        <div className="flex items-center justify-between px-2">
          <h3 className="text-xs font-black text-slate-500 uppercase tracking-[0.4em]">Explorar Especialidades</h3>
          <span className="text-[10px] text-cyan-500 font-bold uppercase tracking-widest">+100 Libros por Categoría</span>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
          {niches.slice(0, 8).map((niche, index) => {
            const style = nicheStyles[niche.name] || nicheStyles['DEFAULT'];
            return (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                key={niche.name}
                className={`group relative p-8 rounded-[2.5rem] bg-gradient-to-br ${style.bg} border ${style.border} hover:scale-[1.02] transition-all duration-500 overflow-hidden`}
              >
                <div className="absolute -right-4 -bottom-4 opacity-5 group-hover:opacity-10 transition-opacity">
                  <style.icon className="w-32 h-32 text-white" />
                </div>
                <div className={`p-4 rounded-2xl bg-slate-900/50 w-fit mb-6 ${style.color} shadow-xl`}>
                  <style.icon className="w-8 h-8" />
                </div>
                <h4 className="text-white font-black text-2xl leading-tight uppercase mb-2">
                  {niche.name.replace(/_/g, ' ')}
                </h4>
                <div className="flex items-center justify-between mt-8">
                  <p className={`text-[10px] font-black ${style.color} opacity-80 tracking-widest uppercase`}>{niche.count} ACTIVOS</p>
                  <button 
                    onClick={() => {
                      const q = `Dame los mejores consejos sobre ${niche.name.replace(/_/g, ' ')}`;
                      setQuery(q);
                      handleSearch(undefined, q);
                    }}
                    className="p-3 bg-white/10 hover:bg-white text-white hover:text-black rounded-xl transition-all group-hover:shadow-[0_0_20px_rgba(255,255,255,0.2)]"
                  >
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </div>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* RESULTADOS MULTI-SECCIÓN */}
      <AnimatePresence>
        {(loading || results) && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.98, y: 40 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98, y: 40 }}
            className="space-y-12"
          >
            <div className="flex items-center gap-6">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent"></div>
              <div className="flex items-center gap-3">
                <Crown className="w-5 h-5 text-cyan-400" />
                <span className="text-xs font-black text-cyan-400 uppercase tracking-[0.5em]">Reporte de Inteligencia Diamond</span>
              </div>
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent"></div>
            </div>

            {loading ? (
              <div className="glass-panel rounded-[4rem] p-32 flex flex-col items-center justify-center gap-10 border-cyan-500/20 bg-cyan-500/[0.02]">
                <div className="relative">
                  <div className="w-32 h-32 border-b-4 border-cyan-500 rounded-full animate-spin"></div>
                  <Database className="absolute inset-0 m-auto w-10 h-10 text-cyan-500 animate-pulse" />
                </div>
                <div className="text-center">
                  <p className="text-3xl font-black text-white uppercase tracking-widest mb-3">Extrayendo Sabiduría</p>
                  <p className="text-slate-500 font-mono text-lg animate-pulse">Cruzando bibliografía de +100 libros en tiempo real...</p>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
                {/* SÍNTESIS PRINCIPAL */}
                <div className="lg:col-span-8 glass-panel rounded-[3.5rem] overflow-hidden border-cyan-500/30 bg-slate-900/40 relative">
                  <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none">
                    <Sparkles className="w-64 h-64 text-cyan-400" />
                  </div>
                  
                  <div className="bg-cyan-500/10 px-12 py-8 border-b border-cyan-500/20 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <Lightbulb className="w-8 h-8 text-cyan-400" />
                      <h4 className="font-black text-white text-xl uppercase tracking-wider">Síntesis Maestra</h4>
                    </div>
                    <div className="flex gap-3">
                       <button 
                         onClick={() => {
                           const blob = new Blob([results?.answer], { type: 'text/plain' });
                           const url = URL.createObjectURL(blob);
                           const a = document.createElement('a');
                           a.href = url;
                           a.download = `Sintesis_Diamond_${new Date().toISOString().split('T')[0]}.txt`;
                           a.click();
                         }}
                         className="p-3 bg-white/5 hover:bg-cyan-500 text-white hover:text-black rounded-xl transition-all shadow-xl flex items-center gap-2 group"
                        >
                          <Share2 className="w-5 h-5" />
                          <span className="text-[10px] font-black hidden group-hover:block uppercase tracking-widest">EXPORTAR</span>
                       </button>
                    </div>
                  </div>
                  
                  <div className="p-12">
                    <div className="whitespace-pre-wrap text-2xl font-medium text-slate-100 leading-relaxed drop-shadow-sm">
                      {results?.answer}
                    </div>
                    
                    <div className="mt-16 pt-8 border-t border-white/5 flex flex-wrap gap-4">
                       <button className="bg-cyan-500 text-black font-black px-10 py-4 rounded-2xl flex items-center gap-3 hover:bg-cyan-400 transition-all active:scale-95 shadow-2xl">
                          <Target className="w-5 h-5" /> APLICAR ESTE CONOCIMIENTO
                       </button>
                       <button className="bg-white/5 border border-white/10 text-white font-black px-10 py-4 rounded-2xl flex items-center gap-3 hover:bg-white/10 transition-all active:scale-95 shadow-xl">
                          <Eye className="w-5 h-5" /> VER DIAGRAMA DE FLUJO
                       </button>
                    </div>
                  </div>
                </div>

                {/* COLUMNA DE FUENTES */}
                <div className="lg:col-span-4 space-y-8">
                  <div className="glass-panel rounded-[3rem] p-10 border-white/10 bg-slate-900/60 shadow-2xl">
                    <div className="flex items-center gap-4 mb-8">
                      <div className="p-3 bg-purple-500/20 rounded-xl">
                        <FileText className="w-6 h-6 text-purple-400" />
                      </div>
                      <h4 className="font-black text-white text-lg uppercase tracking-widest">Fuentes de Verdad</h4>
                    </div>
                    
                    <div className="space-y-4">
                      {results?.sources?.map((source: string, i: number) => (
                        <div key={i} className="group relative">
                          <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-2xl blur opacity-0 group-hover:opacity-20 transition-all"></div>
                          <div className="relative flex flex-col p-5 bg-white/5 border border-white/5 rounded-2xl hover:border-purple-500/30 transition-all">
                            <div className="flex items-start justify-between gap-3 mb-3">
                              <div className="flex items-center gap-3 overflow-hidden">
                                <BookOpen className="w-5 h-5 text-purple-400 flex-shrink-0" />
                                <span className="text-xs font-black text-white truncate uppercase tracking-tight">{source}</span>
                              </div>
                              <span className="text-[9px] font-black bg-purple-500/20 text-purple-400 px-2 py-1 rounded border border-purple-500/20">98% Match</span>
                            </div>
                            
                            <div className="flex gap-2 mt-2">
                               <button 
                                 onClick={() => setPreviewFile(source)}
                                 className="flex-1 bg-white/5 hover:bg-purple-500 text-[10px] font-black text-slate-300 hover:text-white py-2 rounded-lg border border-white/5 transition-all flex items-center justify-center gap-2"
                                >
                                  <Eye className="w-3 h-3" /> ABRIR
                               </button>
                               <button 
                                 onClick={() => handleDownload(source)}
                                 className="flex-1 bg-white/5 hover:bg-cyan-500 text-[10px] font-black text-slate-300 hover:text-white py-2 rounded-lg border border-white/5 transition-all flex items-center justify-center gap-2"
                                >
                                  <Download className="w-3 h-3" /> BAJAR
                               </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
