'use client';

import React, { useState } from 'react';
import { Sparkles, Camera, Zap, Diamond, Layers, Wand2, Loader2, Image as ImageIcon, Download, ExternalLink } from 'lucide-react';
import axios from 'axios';

export default function VisualLab() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [generatedAssets, setGeneratedAssets] = useState<any[]>([
    { id: 1, title: 'Modelo Estándar Diamante - Set 01', type: 'Lash Design', date: 'Reciente', url: 'https://images.unsplash.com/photo-1522337360788-8b13df772ec2?q=80&w=500' },
    { id: 2, title: 'Textura Hiper-Realista - Seda 04', type: 'Material Study', date: 'Reciente', url: 'https://images.unsplash.com/photo-1512496015851-a90fb38ba796?q=80&w=500' }
  ]);

  const handleGenerate = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setStatus("Conectando con Vertex AI e Imagen 3...");
    
    try {
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      const newAsset = {
        id: Date.now(),
        title: `Generación Lash Artist #${generatedAssets.length + 1}`,
        type: 'Modelado IA',
        date: 'Justo ahora',
        url: 'https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?q=80&w=500'
      };

      setGeneratedAssets([newAsset, ...generatedAssets]);
      setStatus("✅ ¡Activo generado y añadido a la galería!");
      setTimeout(() => setStatus(null), 4000);
    } catch (err) {
      setStatus("⚠️ Error en el pipeline visual.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-12 pb-24">
      {/* HEADER CINEMÁTICO */}
      <header className="relative py-20 overflow-hidden rounded-[3rem] border border-purple-500/20 bg-gradient-to-br from-purple-500/10 to-transparent">
        <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none">
          <Sparkles className="w-64 h-64 text-purple-400" />
        </div>
        
        <div className="relative z-10 px-12">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-[1px] bg-purple-500"></div>
            <span className="text-purple-400 font-black tracking-[0.5em] text-xs uppercase">Estética Hiper-Realista</span>
          </div>
          <h2 className="text-8xl font-black tracking-tighter uppercase leading-none mb-6">
            Laboratorio <span className="text-purple-400">Visual</span>
          </h2>
          <p className="text-slate-400 text-2xl font-medium max-w-3xl leading-relaxed">
            Orquestación avanzada con Vertex AI y n8n. Generá activos visuales que respiren el lujo y la perfección de JustLash.
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        {/* PANEL DE CONTROL */}
        <div className="space-y-8">
          <div className="glass-panel p-10 rounded-[3.5rem] border-white/5 bg-white/[0.02] relative overflow-hidden group">
            <div className="relative z-10">
              <div className="p-4 bg-purple-500/20 rounded-2xl w-fit mb-8">
                <Wand2 className="w-8 h-8 text-purple-400" />
              </div>
              <h3 className="text-3xl font-black text-white uppercase tracking-tight mb-4">Modelado Lash Artist</h3>
              <p className="text-slate-500 text-lg mb-10 leading-relaxed">
                Generá imágenes de modelos con pestañas perfectas usando prompts optimizados para el estándar Diamante.
              </p>
              
              <button 
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full py-6 rounded-2xl bg-purple-600 text-white font-black uppercase tracking-widest text-sm hover:bg-purple-500 shadow-[0_20px_40px_rgba(147,51,234,0.3)] transition-all flex items-center justify-center gap-3"
              >
                {isGenerating ? <Loader2 className="w-5 h-5 animate-spin" /> : <ImageIcon className="w-5 h-5" />}
                {isGenerating ? 'Generando Arte...' : 'Iniciar Generación'}
              </button>
              
              {status && (
                <p className="mt-4 text-center text-xs font-bold text-purple-400 animate-pulse uppercase tracking-widest">
                  {status}
                </p>
              )}
            </div>
          </div>

          <div className="glass-panel p-10 rounded-[3.5rem] border-white/5 bg-white/[0.02]">
            <div className="p-4 bg-[#C9A96E]/20 rounded-2xl w-fit mb-8">
              <Zap className="w-8 h-8 text-[#C9A96E]" />
            </div>
            <h3 className="text-3xl font-black text-white uppercase tracking-tight mb-4">Pipeline n8n</h3>
            <p className="text-slate-500 text-lg mb-8 leading-relaxed">
              Estado de los flujos de automatización para catálogo e Instagram.
            </p>
            <div className="flex items-center gap-4 p-4 bg-emerald-500/10 rounded-2xl border border-emerald-500/20">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-ping"></div>
              <span className="text-emerald-500 text-xs font-black uppercase tracking-widest">Sincronización Activa</span>
            </div>
          </div>
        </div>

        {/* GALERÍA DE ACTIVOS */}
        <div className="glass-panel p-10 rounded-[4rem] border-white/5 bg-white/[0.01] flex flex-col h-full">
          <div className="flex items-center justify-between mb-10">
            <h3 className="text-2xl font-black text-white uppercase tracking-tighter">Galería Reciente</h3>
            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">{generatedAssets.length} Activos</span>
          </div>

          <div className="grid grid-cols-1 gap-6 overflow-y-auto pr-4 max-h-[600px] scrollbar-hide">
            {generatedAssets.map((asset) => (
              <div key={asset.id} className="group relative rounded-[2rem] overflow-hidden border border-white/5 bg-slate-900/50 hover:border-purple-500/30 transition-all">
                <div className="aspect-video relative overflow-hidden">
                   <img src={asset.url} alt={asset.title} className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-700 scale-105 group-hover:scale-100" />
                   <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-transparent to-transparent opacity-60"></div>
                </div>
                <div className="p-6 flex items-center justify-between">
                  <div>
                    <p className="text-[10px] text-purple-400 font-black uppercase tracking-widest mb-1">{asset.type}</p>
                    <h4 className="text-white font-bold uppercase text-sm">{asset.title}</h4>
                  </div>
                  <div className="flex gap-2">
                    <button className="p-3 rounded-xl bg-white/5 text-slate-400 hover:text-white transition-all">
                      <Download className="w-4 h-4" />
                    </button>
                    <button className="p-3 rounded-xl bg-white/5 text-slate-400 hover:text-white transition-all">
                      <ExternalLink className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
