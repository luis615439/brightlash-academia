'use client';

import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Send, RotateCcw, User, Bot, Sparkles, CheckCircle2, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  id: string;
  role: 'user' | 'agent';
  content: string;
  agentName?: string;
  timestamp: string;
}

export default function SalesSimulator() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [leadId, setLeadId] = useState('');
  const [agentStatus, setAgentStatus] = useState({
    name: 'Esperando Lead...',
    type: 'offline',
    state: 'idle'
  });
  
  const [availableScripts, setAvailableScripts] = useState<any[]>([]);
  const [selectedScript, setSelectedScript] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Cargar scripts y generar Lead ID al inicio
  useEffect(() => {
    const id = `lead_${Math.random().toString(36).substr(2, 9)}`;
    setLeadId(id);

    const fetchScripts = async () => {
      try {
        const backendUrl = window.location.hostname === 'localhost' 
          ? 'http://localhost:8000' 
          : 'https://upset-sloths-listen.loca.lt';
        const res = await fetch(`${backendUrl}/api/portal/resources`);
        const data = await res.json();
        setAvailableScripts(data.scripts || []);
      } catch (err) {
        console.error('Error fetching scripts for simulator:', err);
      }
    };
    fetchScripts();
  }, []);

  // Auto-scroll al final del chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const backendUrl = window.location.hostname === 'localhost' 
        ? 'http://localhost:8000' 
        : 'https://upset-sloths-listen.loca.lt';
      
      const response = await fetch(`${backendUrl}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lead_id: leadId,
          message: inputValue,
          script_context: selectedScript, // Inyectar el script seleccionado
          dry_run: true
        })
      });

      if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
      
      const data = await response.json();

      if (data.message) {
        const agentMsg: Message = {
          id: (Date.now() + 1).toString(),
          role: 'agent',
          content: data.message,
          agentName: data.agent_name,
          timestamp: new Date().toLocaleTimeString()
        };
        setMessages(prev => [...prev, agentMsg]);
        setAgentStatus({
          name: data.agent_name,
          type: data.agent_type,
          state: data.state_after
        });
      }
    } catch (error: any) {
      console.error("Error en simulación:", error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'agent',
        content: `⚠️ Error de conexión: ${error.message}. Por favor, asegurate de que el backend está corriendo.`,
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    if (window.confirm("¿Estás seguro de reiniciar la simulación?")) {
      try {
        const backendUrl = window.location.hostname === 'localhost' 
          ? 'http://localhost:8000' 
          : 'https://upset-sloths-listen.loca.lt';
        await fetch(`${backendUrl}/api/simulate/reset?lead_id=${leadId}`, { method: 'POST' });
        
        // Limpieza total de estados
        setMessages([]);
        setAgentStatus({ name: 'Esperando Lead...', type: 'offline', state: 'idle' });
        setInputValue('');
        setSelectedScript(null);
        
        // Regenerar identidad para evitar persistencia de bucles
        const newId = `lead_${Math.random().toString(36).substr(2, 9)}`;
        setLeadId(newId);
        
      } catch (error) {
        console.error("Error al reiniciar:", error);
      }
    }
  };

  return (
    <div className="space-y-12 pb-24">
      <header className="relative py-20 overflow-hidden rounded-[3rem] border border-[#C9A96E]/20 bg-gradient-to-br from-[#C9A96E]/10 to-transparent">
        <div className="relative z-10 px-12">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-[1px] bg-[#C9A96E]"></div>
            <span className="text-[#C9A96E] font-black tracking-[0.5em] text-xs uppercase">Simulación Nativa</span>
          </div>
          <h2 className="text-8xl font-black tracking-tighter uppercase leading-none mb-6">
            Simulador <span className="text-[#C9A96E]">Diamante</span>
          </h2>
          <p className="text-slate-400 text-2xl font-medium max-w-3xl leading-relaxed">
            Entrenamiento táctico con IA. Usá los guiones maestros de JustLash para dominar el cierre.
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        <div className="lg:col-span-1 space-y-6">
          {/* SCRIPT SELECTOR */}
          <div className="glass-panel p-8 rounded-[2rem] border-[#C9A96E]/20 bg-[#C9A96E]/5">
            <h3 className="text-[#C9A96E] font-black text-xs uppercase tracking-widest mb-6 flex items-center gap-2">
              <Sparkles className="w-4 h-4" /> Metodología de Venta
            </h3>
            
            <div className="space-y-2">
              {availableScripts.length > 0 ? (
                <div className="relative">
                  <select 
                    value={selectedScript || ''}
                    onChange={(e) => setSelectedScript(e.target.value)}
                    className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white text-xs font-bold appearance-none focus:outline-none focus:border-[#C9A96E]/50 transition-all"
                  >
                    <option value="">Metodología Estándar</option>
                    {availableScripts.map((script, idx) => (
                      <option key={idx} value={script.name}>
                        {script.title || script.name.replace('.md', '').replace(/_/g, ' ')}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#C9A96E] pointer-events-none" />
                </div>
              ) : (
                <p className="text-slate-500 text-[10px] font-bold">Cargando guiones maestros...</p>
              )}
              <p className="text-[10px] text-slate-500 font-medium leading-tight mt-3">
                Seleccioná un guion para que la IA lo use como base en esta sesión.
              </p>
            </div>
          </div>

          <div className="glass-panel p-8 rounded-[2rem] border-white/5 bg-white/[0.02]">
            <h3 className="text-[#C9A96E] font-black text-xs uppercase tracking-widest mb-6 flex items-center gap-2">
              <Bot className="w-4 h-4" /> Estado del Agente
            </h3>
            
            <div className="space-y-4">
              <div>
                <p className="text-slate-500 text-[10px] uppercase font-bold tracking-tighter mb-1">Agente Activo</p>
                <p className="text-white font-bold text-lg">{agentStatus.name}</p>
              </div>
              
              <div>
                <p className="text-slate-500 text-[10px] uppercase font-bold tracking-tighter mb-1">Estado del Funnel</p>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#C9A96E]/10 border border-[#C9A96E]/20 text-[#C9A96E] text-[10px] font-black uppercase">
                  {agentStatus.state}
                </div>
              </div>

              <button 
                onClick={handleReset}
                className="w-full mt-8 flex items-center justify-center gap-3 px-6 py-4 rounded-xl border border-white/10 text-white/50 hover:text-white hover:bg-white/5 transition-all text-xs font-black uppercase tracking-widest"
              >
                <RotateCcw className="w-4 h-4" /> Reiniciar
              </button>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3 glass-panel rounded-[3rem] border-white/5 bg-white/[0.01] overflow-hidden h-[700px] flex flex-col">
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 space-y-6 scrollbar-hide">
            {messages.length === 0 && !isLoading && (
              <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-30">
                <MessageSquare className="w-16 h-16 text-[#C9A96E]" />
                <p className="text-white font-medium">Iniciá la conversación como si fueras una clienta...</p>
                <p className="text-slate-500 text-xs">Ejemplo: "Hola, me interesa el curso de pestañas"</p>
              </div>
            )}

            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[80%] space-y-1 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                    <div className="flex items-center gap-2 mb-1">
                      {msg.role === 'agent' && <Bot className="w-3 h-3 text-[#C9A96E]" />}
                      <span className="text-[10px] font-black uppercase tracking-widest text-slate-500">
                        {msg.role === 'user' ? 'Tú' : msg.agentName || 'Sistema'} • {msg.timestamp}
                      </span>
                    </div>
                    <div className={`p-5 rounded-3xl text-sm leading-relaxed ${
                      msg.role === 'user' 
                        ? 'bg-gradient-to-br from-[#C9A96E] to-[#b08d4d] text-[#0a0906] font-bold rounded-tr-none shadow-lg shadow-[#C9A96E]/10' 
                        : 'bg-white/5 text-slate-200 border border-white/5 rounded-tl-none'
                    }`}>
                      {msg.content.split('\n').map((line, i) => (
                        <React.Fragment key={i}>{line}<br /></React.Fragment>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {isLoading && (
              <div className="bg-white/5 p-5 rounded-3xl border border-white/5 inline-flex items-center gap-3">
                <div className="flex gap-1">
                  <span className="w-1.5 h-1.5 bg-[#C9A96E] rounded-full animate-bounce"></span>
                  <span className="w-1.5 h-1.5 bg-[#C9A96E] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-1.5 h-1.5 bg-[#C9A96E] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
                <span className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Escribiendo...</span>
              </div>
            )}
          </div>

          <form onSubmit={handleSend} className="p-6 bg-white/[0.02] border-t border-white/5 flex items-center gap-4">
            <input 
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Escribí tu mensaje aquí..."
              className="flex-1 bg-white/5 border border-white/10 rounded-2xl px-6 py-4 text-white text-sm focus:outline-none focus:border-[#C9A96E]/50 transition-all placeholder:text-white/20 font-medium"
              disabled={isLoading}
            />
            <button 
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              className="p-4 rounded-2xl bg-[#C9A96E] text-[#0a0906] hover:scale-105 active:scale-95 disabled:opacity-50 disabled:hover:scale-100 transition-all shadow-lg shadow-[#C9A96E]/20"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
