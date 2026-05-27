import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Zap, 
  BookOpen, 
  MessageSquare, 
  Sparkles, 
  ArrowLeft, 
  Diamond,
  ChevronRight,
  TrendingUp,
  Target,
  FileText,
  Share2,
  Download,
  X
} from 'lucide-react';
import SalesSimulator from './SalesSimulator';

export default function SalesPortal() {
  const [resources, setResources] = useState<{lessons: any[], scripts: any[]}>({ lessons: [], scripts: [] });
  const [selectedContent, setSelectedContent] = useState<string | null>(null);
  const [activeFilename, setActiveFilename] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchResources = async () => {
      try {
        const backendUrl = window.location.hostname === 'localhost' 
          ? 'http://localhost:8000' 
          : 'https://upset-sloths-listen.loca.lt';
        const res = await axios.get(`${backendUrl}/api/portal/resources`);
        setResources(res.data);
      } catch (err) {
        console.error('Error fetching portal resources:', err);
      }
    };
    fetchResources();
  }, []);

  const loadContent = async (type: string, name: string) => {
    if (loading) return;
    setLoading(true);
    setActiveFilename(name);
    setSelectedContent(null);
    
    try {
      const backendUrl = window.location.hostname === 'localhost' 
        ? 'http://localhost:8000' 
        : 'https://upset-sloths-listen.loca.lt';
      const res = await axios.get(`${backendUrl}/api/portal/content?path_type=${type}&filename=${encodeURIComponent(name)}`);
      
      if (res.data && res.data.content) {
        setSelectedContent(res.data.content);
      } else {
        setSelectedContent("⚠️ El archivo está vacío o no se pudo leer el contenido.");
      }
    } catch (err: any) {
      console.error("Error loading content:", err);
      const detail = err.response?.data?.detail || "Asegurate de que el backend esté corriendo y el archivo exista.";
      setSelectedContent(`❌ Error de conexión: ${detail}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0906] text-[#F8F5F0] space-y-12 pb-24 px-8 scroll-smooth">
      {/* HEADER CINEMÁTICO */}
      <header className="relative py-20 overflow-hidden rounded-[3rem] border border-[#C9A96E]/20 bg-gradient-to-br from-[#C9A96E]/10 to-transparent px-12">
        <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none">
          <Diamond className="w-64 h-64 text-[#C9A96E]" />
        </div>
        
        <div className="relative z-10">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-[1px] bg-[#C9A96E]"></div>
            <span className="text-[#C9A96E] font-black tracking-[0.5em] text-xs uppercase">Estándar Diamante</span>
          </div>
          <h2 className="text-8xl font-black tracking-tighter uppercase leading-none mb-6">
            Portal de <span className="text-[#C9A96E]">Cristal</span>
          </h2>
          <p className="text-slate-400 text-2xl font-medium max-w-3xl leading-relaxed">
            Sabiduría destilada para Arquitectas de la Belleza. Contenido de alto impacto para dominar el arte de la persuasión y el cierre.
          </p>
        </div>
      </header>

      {/* APPS INCRUSTRADAS (CENTRO DE MANDO) */}
      <section className="mb-20">
        <div className="flex items-center gap-4 mb-10">
          <div className="w-8 h-[1px] bg-[#C9A96E]"></div>
          <h3 className="text-xl font-black text-white uppercase tracking-widest">Sistemas Operativos</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div onClick={() => {
            const el = document.getElementById('simulator');
            el?.scrollIntoView({ behavior: 'smooth' });
          }} className="p-10 rounded-[3rem] bg-gradient-to-br from-[#C9A96E]/20 to-transparent border border-[#C9A96E]/30 cursor-pointer hover:scale-[1.02] transition-all group">
            <div className="flex items-center justify-between mb-6">
              <MessageSquare className="w-10 h-10 text-[#C9A96E]" />
              <div className="px-4 py-1 rounded-full bg-[#C9A96E]/10 border border-[#C9A96E]/20 text-[#C9A96E] text-[10px] font-black uppercase tracking-widest">Activo</div>
            </div>
            <h4 className="text-3xl font-black text-white uppercase mb-2">Simulador Diamante</h4>
            <p className="text-slate-400 font-medium">Entrenamiento de ventas con IA en tiempo real.</p>
          </div>

          <div className="p-10 rounded-[3rem] bg-gradient-to-br from-purple-500/20 to-transparent border border-purple-500/30 cursor-default opacity-80">
            <div className="flex items-center justify-between mb-6">
              <Zap className="w-10 h-10 text-purple-400" />
              <div className="px-4 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-[10px] font-black uppercase tracking-widest">Procesando Batch 04</div>
            </div>
            <h4 className="text-3xl font-black text-white uppercase mb-2">Destilador Maestro</h4>
            <p className="text-slate-400 font-medium">40 / 416 Activos de conocimiento procesados.</p>
          </div>
        </div>
      </section>

      {/* CUERPO DEL PORTAL (LISTA + VISOR) */}
      <div className="flex flex-col lg:flex-row gap-8 items-start min-h-[900px]">
        {/* LISTADO DE RECURSOS (COLUMNA IZQUIERDA) */}
        <aside className="w-full lg:w-1/3 xl:w-1/4 space-y-8 sticky top-8">
          <section>
            <div className="flex items-center justify-between mb-8 px-4">
              <h3 className="text-sm font-black text-slate-500 uppercase tracking-[0.4em] flex items-center gap-3">
                <BookOpen className="w-4 h-4 text-[#C9A96E]" />
                Micro-Lecciones
              </h3>
            </div>
            <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2 scrollbar-hide">
              {resources.lessons.map((lesson, i) => (
                <button
                  key={i}
                  onClick={() => loadContent('leccion', lesson.name)}
                  className={`w-full group text-left p-5 rounded-[1.5rem] border transition-all duration-300 ${
                    activeFilename === lesson.name 
                    ? 'bg-[#C9A96E] border-[#C9A96E] text-[#0a0906] shadow-[0_0_30px_rgba(201,169,110,0.3)]' 
                    : 'bg-white/5 border-white/5 hover:border-[#C9A96E]/30 hover:bg-white/[0.07] text-white'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-[9px] font-black uppercase tracking-widest ${activeFilename === lesson.name ? 'text-[#0a0906]/60' : 'text-slate-500'}`}>Lección {i+1}</span>
                    <ChevronRight className={`w-4 h-4 ${activeFilename === lesson.name ? 'text-[#0a0906]' : 'text-slate-700'}`} />
                  </div>
                  <h4 className="text-sm font-black uppercase truncate">{lesson.name.replace('LECCION_', '').split('.')[0].replace(/_/g, ' ')}</h4>
                </button>
              ))}
            </div>
          </section>

          <section>
            <div className="flex items-center justify-between mb-8 px-4">
              <h3 className="text-sm font-black text-slate-500 uppercase tracking-[0.4em] flex items-center gap-3">
                <MessageSquare className="w-4 h-4 text-purple-400" />
                Guiones Premium
              </h3>
            </div>
            <div className="space-y-4">
              {resources.scripts.map((script, i) => (
                <button
                  key={i}
                  onClick={() => loadContent('guion', script.name)}
                  className={`w-full group text-left p-6 rounded-[2rem] border transition-all duration-500 ${
                    activeFilename === script.name 
                    ? 'bg-purple-600 border-purple-600 text-white shadow-[0_0_40px_rgba(147,51,234,0.2)]' 
                    : 'bg-white/5 border-white/5 hover:border-purple-500/30 hover:bg-white/[0.07]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <Sparkles className={`w-4 h-4 ${activeFilename === script.name ? 'text-white' : 'text-purple-500'}`} />
                    <ChevronRight className={`w-4 h-4 ${activeFilename === script.name ? 'text-white' : 'text-slate-700'}`} />
                  </div>
                  <h4 className="text-lg font-black leading-tight uppercase truncate">{script.name.replace('GUION_', '').split('.')[0]}</h4>
                </button>
              ))}
            </div>
          </section>
        </aside>

        {/* VISUALIZADOR DE CONTENIDO (COLUMNA DERECHA) */}
        <main className="w-full lg:w-2/3 xl:w-3/4 h-[800px]">
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="glass-panel p-16 rounded-[4rem] border-[#C9A96E]/20 bg-white/[0.02] h-full flex flex-col items-center justify-center"
              >
                <div className="w-20 h-20 border-4 border-[#C9A96E]/20 border-t-[#C9A96E] rounded-full animate-spin mb-8"></div>
                <p className="text-[#C9A96E] font-black uppercase tracking-widest animate-pulse">Descifrando Sabiduría...</p>
              </motion.div>
            ) : selectedContent ? (
              <motion.div
                key={activeFilename}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="glass-panel rounded-[4rem] border-[#C9A96E]/20 bg-white/[0.01] h-full relative overflow-hidden flex flex-col"
              >
                <div className="p-8 border-b border-white/5 bg-white/[0.02] flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-[#C9A96E]/10 rounded-xl">
                      <FileText className="w-6 h-6 text-[#C9A96E]" />
                    </div>
                    <div>
                      <h4 className="text-xl font-black text-white uppercase tracking-tight truncate max-w-md">
                        {activeFilename?.replace('.md', '').replace(/_/g, ' ')}
                      </h4>
                      <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Activo Digital Diamond • JustLash</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <button className="p-3 rounded-xl bg-white/5 text-slate-400 hover:text-white transition-all">
                      <Share2 className="w-5 h-5" />
                    </button>
                    <button className="p-3 rounded-xl bg-white/5 text-slate-400 hover:text-white transition-all">
                      <Download className="w-5 h-5" />
                    </button>
                    <button 
                      onClick={() => { setSelectedContent(null); setActiveFilename(null); }}
                      className="p-3 rounded-xl bg-white/5 text-slate-400 hover:text-white transition-all"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                <div className="flex-1 overflow-y-auto p-16 scrollbar-hide">
                  <div className="prose prose-invert max-w-none">
                    <div className="text-2xl text-[#F8F5F0] leading-[1.8] font-medium whitespace-pre-wrap selection:bg-[#C9A96E] selection:text-[#0a0906]">
                      {selectedContent}
                    </div>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="glass-panel p-16 rounded-[4rem] border-white/5 bg-white/[0.01] h-full flex flex-col items-center justify-center text-center"
              >
                <div className="w-32 h-32 bg-white/5 rounded-full flex items-center justify-center mb-10 border border-white/10">
                  <Diamond className="w-16 h-16 text-slate-700 animate-pulse" />
                </div>
                <h3 className="text-3xl font-black text-white uppercase tracking-tighter mb-4">Iniciando Protocolo de Lectura</h3>
                <p className="text-slate-500 text-lg max-w-md font-medium">
                  Seleccioná una pieza de sabiduría del panel izquierdo para comenzar la inmersión en el estándar Diamante.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* SECCIÓN DEL SIMULADOR (ANCHOR) */}
      <section id="simulator" className="pt-32 border-t border-white/5">
        <div className="flex items-center gap-4 mb-16">
          <div className="w-12 h-[1px] bg-[#C9A96E]"></div>
          <h3 className="text-4xl font-black text-white uppercase tracking-tighter">Entorno de Simulación <span className="text-[#C9A96E]">Diamante</span></h3>
        </div>
        
        <div className="glass-panel rounded-[4rem] border-[#C9A96E]/20 bg-white/[0.01] overflow-hidden shadow-[0_0_100px_rgba(201,169,110,0.05)]">
           <SalesSimulator />
        </div>
      </section>
    </div>
  );
}
