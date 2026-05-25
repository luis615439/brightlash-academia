import React from 'react';
import { Sun, Moon } from 'lucide-react';

interface ThemeToggleProps {
  theme: 'light' | 'dark';
  onToggle: (theme: 'light' | 'dark') => void;
}

export default function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  return (
    <button
      onClick={() => onToggle(theme === 'light' ? 'dark' : 'light')}
      className="fixed top-4 right-4 z-50 p-3 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-150 rounded-2xl shadow-md border border-gray-100 dark:border-gray-700/60 hover:shadow-lg transition-all duration-300 active:scale-95"
      aria-label="Toggle Theme"
    >
      {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
    </button>
  );
}
