import React, { useState, useEffect } from 'react';

const Dashboard = () => {
  const [activeModule, setActiveModule] = useState('HOME');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  // Backend Integration State
  const [libraryData, setLibraryData] = useState({ pending: 10, status: 'idle' });
  const [agentLogs, setAgentLogs] = useState([]);
  const [agentInput, setAgentInput] = useState('');

  // Simulación de carga de endpoints al montar los módulos
  useEffect(() => {
    if (activeModule === 'LIBRARY') {
      fetchLibraryData();
    } else if (activeModule === 'AGENT') {
      fetchAgentStatus();
    }
  }, [activeModule]);

  const fetchLibraryData = async () => {
    try {
      // Endpoint Connection (Simulated/Mapped to local API)
      setLibraryData({ pending: 10, status: 'loading' });
      setTimeout(() => {
        setLibraryData({ pending: 10, status: 'connected', timestamp: Date.now() });
      }, 600);
    } catch (err) {
      console.error("API Error:", err);
    }
  };

  const fetchAgentStatus = async () => {
    try {
      // Endpoint Connection (Simulated/Mapped to local API)
      setAgentLogs([{ id: 1, type: 'system', text: 'Lead cualificado detectado. Iniciando script de cierre high-ticket.' }]);
    } catch (err) {
      console.error("API Error:", err);
    }
  };

  const handleAgentSend = () => {
    if (!agentInput) return;
    setAgentLogs(prev => [...prev, { id: Date.now(), type: 'user', text: agentInput }]);
    setAgentInput('');
    
    // Simulate backend response
    setTimeout(() => {
      setAgentLogs(prev => [...prev, { id: Date.now()+1, type: 'system', text: '[OS Executing] Comando procesado. Actualizando CRM.' }]);
    }, 800);
  };

  const LibraryModule = () => (
    <div id="library-module" className="animate-fade-in flex flex-col h-full bg-slate-900/50 rounded-xl border border-slate-700/50 p-6 shadow-[inset_0_0_20px_rgba(0,0,0,0.5)]">
      <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-teal-200 mb-4">Biblioteca Digital Semántica</h2>
      <div className="flex-1 overflow-y-auto space-y-4">
        <div className="bg-slate-800/80 p-4 rounded-lg border border-slate-700/60 flex items-center justify-between">
          <div>
            <h3 className="text-white font-medium">Lote 11: Auditoría e Ingesta</h3>
            <p className="text-xs text-slate-400 mt-1">
              {libraryData.status === 'connected' ? `${libraryData.pending} libros pendientes listos para destilación.` : 'Conectando al motor de ingesta...'}
            </p>
          </div>
          <button 
            className="px-4 py-2 bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 rounded-md text-sm font-semibold hover:bg-emerald-600/40 transition-colors"
            disabled={libraryData.status !== 'connected'}
          >
            {libraryData.status === 'connected' ? 'Iniciar Ingesta API' : 'Sincronizando...'}
          </button>
        </div>
      </div>
    </div>
  );

  const AgentModule = () => (
    <div id="agent-module" className="animate-fade-in flex flex-col h-full bg-slate-900/50 rounded-xl border border-slate-700/50 p-6 shadow-[inset_0_0_20px_rgba(0,0,0,0.5)]">
      <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-purple-200 mb-4">JustLash OS - Agente Vendedor</h2>
      <div className="flex-1 flex flex-col gap-4">
        <div className="flex-1 bg-slate-950/80 rounded-lg border border-slate-800 p-4 overflow-y-auto flex flex-col gap-3">
          {agentLogs.map(log => (
            <div key={log.id} className="flex gap-3">
               <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs border ${log.type === 'system' ? 'bg-indigo-500/20 text-indigo-400 border-indigo-500/50' : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50'}`}>
                 {log.type === 'system' ? 'OS' : 'YO'}
               </div>
               <div className="bg-slate-800/60 p-3 rounded-lg text-sm text-slate-200 border border-slate-700/30">
                 {log.text}
               </div>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <input 
            type="text" 
            placeholder="Comando API para el agente..." 
            value={agentInput}
            onChange={(e) => setAgentInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAgentSend()}
            className="flex-1 bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-2 text-sm text-white focus:outline-none focus:border-indigo-500/50" 
          />
          <button 
            onClick={handleAgentSend}
            className="px-6 py-2 bg-indigo-600 border border-indigo-500 rounded-lg text-white font-semibold shadow-[0_0_15px_rgba(99,102,241,0.4)] hover:shadow-[0_0_25px_rgba(99,102,241,0.6)] transition-all"
          >
            POST Execute
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#05080f] text-slate-200 flex flex-col md:flex-row font-sans overflow-hidden">
      
      {/* Mobile Toggle */}
      <button 
        className="md:hidden fixed top-4 left-4 z-50 p-2 bg-slate-800 rounded-md border border-slate-700"
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
      >
        <span className="text-xl">☰</span>
      </button>

      {/* Titaniumorphism Sidebar */}
      <aside className={`fixed md:relative z-40 w-64 h-screen bg-slate-900/70 backdrop-blur-2xl border-r border-slate-700/50 shadow-[4px_0_24px_rgba(0,0,0,0.6)] flex flex-col transition-transform duration-300 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}`}>
        <div className="p-6 border-b border-slate-800/80">
          <h1 className="text-xl font-bold tracking-widest text-white uppercase drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]">Portal de Cristal</h1>
          <p className="text-[10px] text-slate-500 tracking-widest mt-1">V2.0 LIVE API</p>
        </div>
        
        <nav className="flex-1 py-6 px-4 space-y-2">
          <button 
            id="nav-library"
            onClick={() => setActiveModule('LIBRARY')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${activeModule === 'LIBRARY' ? 'bg-slate-800 border border-slate-600/50 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.1)]' : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'}`}
          >
            📚 Biblioteca Digital
          </button>
          
          <button 
            id="nav-agent"
            onClick={() => setActiveModule('AGENT')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${activeModule === 'AGENT' ? 'bg-slate-800 border border-slate-600/50 text-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.1)]' : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'}`}
          >
            🤖 JustLash OS
          </button>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 h-screen overflow-hidden bg-gradient-to-br from-[#05080f] to-[#0a0f18] p-4 md:p-8 flex flex-col relative">
        <header className="flex justify-between items-center mb-8 pl-12 md:pl-0">
          <h2 className="text-xl text-slate-300 font-semibold tracking-wide">
            {activeModule === 'HOME' ? 'Command Center' : activeModule === 'LIBRARY' ? 'API Módulo de Auditoría' : 'API Módulo de Ventas'}
          </h2>
          <div className="flex items-center gap-3">
             <div className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse"></div>
             <span className="text-xs font-bold text-slate-400 tracking-widest">NEXT.JS API ROUTED</span>
          </div>
        </header>

        <section className="flex-1 relative overflow-hidden">
          {activeModule === 'HOME' && (
            <div className="absolute inset-0 flex items-center justify-center text-center">
              <div>
                <div className="w-24 h-24 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 flex items-center justify-center shadow-[0_0_40px_rgba(0,0,0,0.5)] transform rotate-45">
                   <div className="w-12 h-12 bg-gradient-to-br from-slate-400 to-white opacity-20 -rotate-45"></div>
                </div>
                <h3 className="text-2xl font-light text-slate-400 tracking-widest">Selecciona un endpoint visual</h3>
              </div>
            </div>
          )}
          {activeModule === 'LIBRARY' && <LibraryModule />}
          {activeModule === 'AGENT' && <AgentModule />}
        </section>
      </main>
    </div>
  );
};

export default Dashboard;
