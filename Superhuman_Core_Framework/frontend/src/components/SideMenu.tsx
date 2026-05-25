import React, { useState } from 'react';
import { Menu, X, LucideIcon } from 'lucide-react';

export interface AppConfig {
  id: string;
  name: string;
  icon: LucideIcon;
}

interface SideMenuProps {
  activeApp: string;
  onSelectApp: (app: string) => void;
  apps: AppConfig[];
}

export default function SideMenu({ activeApp, onSelectApp, apps }: SideMenuProps) {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);

  return (
    <>
      {/* Botón de Hamburguesa para Móvil */}
      <button
        onClick={toggleMenu}
        className="lg:hidden fixed top-4 left-4 z-50 p-3 bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-100 dark:border-gray-700/60 transition-all duration-300"
      >
        {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Menú Lateral */}
      <div
        className={`fixed left-0 top-0 h-full bg-white dark:bg-gray-800 shadow-2xl z-40 transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } lg:translate-x-0 w-64 border-r border-gray-100 dark:border-gray-700/50`}
      >
        {/* Cabecera */}
        <div className="p-6 border-b border-gray-150 dark:border-gray-700/50">
          <h2 className="text-xl font-bold text-gray-850 dark:text-white">
            Superhuman OS
          </h2>
          <p className="text-[11px] text-gray-400 dark:text-gray-400 mt-1 uppercase tracking-wider">
            Core Portal Framework
          </p>
        </div>

        {/* Links de Navegación */}
        <nav className="p-4 space-y-1.5 overflow-y-auto max-h-[calc(100vh-120px)]">
          {apps.map((app) => {
            const Icon = app.icon;
            const isActive = activeApp === app.id;

            return (
              <button
                key={app.id}
                onClick={() => {
                  onSelectApp(app.id);
                  setIsOpen(false);
                }}
                className={`w-full flex items-center gap-3.5 px-4 py-3 rounded-2xl font-semibold text-sm transition-all duration-300 ${
                  isActive
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-blue-500/10'
                    : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-750'
                }`}
              >
                <Icon className="w-4.5 h-4.5" />
                <span>{app.name}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Overlay para Móvil */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/40 backdrop-blur-sm z-30"
          onClick={toggleMenu}
        />
      )}
    </>
  );
}
