import React, { useState, useEffect } from 'react';
import SideMenu, { AppConfig } from './components/SideMenu';
import ThemeToggle from './components/ThemeToggle';
import GenericForm from './components/GenericForm';
import GenericMetricsAuditor from './components/GenericMetricsAuditor';
import { Megaphone, BarChart4 } from 'lucide-react';

type Theme = 'light' | 'dark';

export default function App() {
  const [activeApp, setActiveApp] = useState('generic-form');
  const [theme, setTheme] = useState<Theme>(() => {
    const savedTheme = localStorage.getItem('theme') as Theme;
    return savedTheme || 'light';
  });

  useEffect(() => {
    localStorage.setItem('theme', theme);
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Lista de Aplicaciones registradas
  const apps: AppConfig[] = [
    { id: 'generic-form', name: 'Formulario de Anuncios', icon: Megaphone },
    { id: 'metrics-auditor', name: 'Auditor de Métricas', icon: BarChart4 },
  ];

  const renderApp = () => {
    switch (activeApp) {
      case 'generic-form':
        return <GenericForm />;
      case 'metrics-auditor':
        return <GenericMetricsAuditor />;
      default:
        return <GenericForm />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 transition-colors duration-300">
      {/* Barra de menú lateral */}
      <SideMenu activeApp={activeApp} onSelectApp={setActiveApp} apps={apps} />
      
      {/* Botón de tema claro/oscuro */}
      <ThemeToggle theme={theme} onToggle={setTheme} />
      
      {/* Contenedor principal de la app activa */}
      <div className="lg:ml-64 min-h-screen">
        <div className="w-full h-full animate-fade-in">
          {renderApp()}
        </div>
      </div>
    </div>
  );
}
