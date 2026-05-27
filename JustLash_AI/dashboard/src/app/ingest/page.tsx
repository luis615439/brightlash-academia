'use client';

import React, { useState } from 'react';
import { Upload, FileText, CheckCircle2, AlertCircle, Loader2, Sparkles, Diamond } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

export default function IngestHangar() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setSuccess(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post('http://localhost:8000/api/ingest', formData);
      setSuccess(`¡Éxito! El libro "${file.name}" ha sido clasificado en el nicho: ${res.data.category}`);
      setFile(null); // Limpiar para el siguiente
    } catch (err) {
      console.error(err);
      setError('Error al procesar el archivo. Revisa el formato.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-12">
      <header>
        <div className="flex items-center gap-6 mb-4">
          <div className="p-4 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-2xl shadow-[0_0_30px_rgba(168,85,247,0.3)]">
            <Upload className="w-10 h-10 text-white" />
          </div>
          <h2 className="text-6xl font-black text-white tracking-tight">Hangar de <span className="text-purple-400">Ingesta</span></h2>
        </div>
        <p className="text-slate-400 text-xl font-medium max-w-2xl leading-relaxed">
          Alimenta tu biblioteca con nuevos conocimientos. La IA se encarga de leer, clasificar e indexar todo en segundos.
        </p>
      </header>

      <div className="max-w-4xl mx-auto">
        <AnimatePresence mode="wait">
          {success ? (
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="glass-panel p-16 rounded-[3rem] border-green-500/30 bg-green-500/5 text-center"
            >
              <div className="w-24 h-24 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-8 shadow-[0_0_50px_rgba(34,197,94,0.3)]">
                <CheckCircle2 className="w-12 h-12 text-green-400" />
              </div>
              <h3 className="text-4xl font-black text-white mb-4">¡Ingesta Completada!</h3>
              <p className="text-green-400 text-xl font-medium mb-10">{success}</p>
              <button 
                onClick={() => setSuccess(null)}
                className="bg-white text-black font-black px-12 py-5 rounded-2xl hover:bg-slate-200 transition-all active:scale-95"
              >
                Subir otro libro
              </button>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-panel p-16 rounded-[3.5rem] border-white/10 bg-white/5 relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-8 opacity-10">
                <Diamond className="w-32 h-32 text-white" />
              </div>

              <div className="flex flex-col items-center justify-center border-4 border-dashed border-white/10 rounded-[2.5rem] p-16 hover:border-cyan-500/30 transition-all group relative">
                <input
                  type="file"
                  onChange={handleFileChange}
                  className="absolute inset-0 opacity-0 cursor-pointer z-10"
                />
                
                <div className="p-8 bg-white/5 rounded-3xl mb-8 group-hover:scale-110 group-hover:bg-cyan-500/10 transition-all">
                  {file ? <FileText className="w-16 h-16 text-cyan-400" /> : <Upload className="w-16 h-16 text-slate-600" />}
                </div>

                <p className="text-3xl font-black text-white mb-3">
                  {file ? file.name : 'Arrastra tus libros aquí'}
                </p>
                <p className="text-slate-500 text-lg font-medium">Soporta PDF, DOCX y TXT</p>
              </div>

              <div className="mt-12 flex flex-col items-center">
                <button
                  onClick={handleUpload}
                  disabled={!file || loading}
                  className="w-full bg-white text-black font-black py-8 rounded-[2rem] text-2xl shadow-2xl hover:bg-cyan-400 transition-all disabled:opacity-20 active:scale-[0.98] flex items-center justify-center gap-4"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-8 h-8 animate-spin" />
                      PROCESANDO CON IA...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-8 h-8" />
                      INICIAR INGESTA
                    </>
                  )}
                </button>
                
                {error && (
                  <div className="mt-8 flex items-center gap-3 text-red-400 font-bold bg-red-500/10 px-6 py-3 rounded-xl border border-red-500/20">
                    <AlertCircle className="w-5 h-5" />
                    {error}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
