import React, { useState } from 'react';
import { Send, Sparkles, RefreshCw, AlertTriangle, CheckCircle } from 'lucide-react';

interface AdCopy {
  headline: string;
  body: string;
  cta?: string;
}

export default function GenericForm() {
  const [category, setCategory] = useState<string>('');
  const [targetGoal, setTargetGoal] = useState<string>('');
  const [dailyBudget, setDailyBudget] = useState<string>('');
  const [webhookUrl, setWebhookUrl] = useState<string>('https://tu-servidor-n8n.co/webhook-test/uuid');

  const [adCopy, setAdCopy] = useState<AdCopy | null>(null);
  const [loadingCopy, setLoadingCopy] = useState(false);
  const [loadingSubmit, setLoadingSubmit] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const generateCopy = async () => {
    if (!category || !targetGoal || !dailyBudget) {
      setError('Completa la categoría, objetivo y presupuesto para generar el copy.');
      return;
    }

    setLoadingCopy(true);
    setError(null);
    setAdCopy(null);

    try {
      const response = await fetch('http://localhost:8000/api/generate-copy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          category,
          target_goal: targetGoal,
          daily_budget: parseFloat(dailyBudget),
        }),
      });

      if (!response.ok) {
        throw new Error('Error al generar el copy con la IA.');
      }

      const data = await response.json();
      setAdCopy(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Error de red. Verifica que el backend de FastAPI esté activo en el puerto 8000.');
    } finally {
      setLoadingCopy(false);
    }
  };

  const submitWorkflow = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!category || !targetGoal || !dailyBudget) {
      setError('Completa los campos obligatorios del formulario.');
      return;
    }

    setLoadingSubmit(true);
    setError(null);
    setSuccessMsg(null);

    try {
      // Si no hay copy generado, lo creamos primero transparentemente
      let activeCopy = adCopy;
      if (!activeCopy) {
        const copyResponse = await fetch('http://localhost:8000/api/generate-copy', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            category,
            target_goal: targetGoal,
            daily_budget: parseFloat(dailyBudget),
          }),
        });

        if (!copyResponse.ok) {
          throw new Error('Error al autogenerar el copy con IA.');
        }
        activeCopy = await copyResponse.json();
        setAdCopy(activeCopy);
      }

      const payload = {
        userInput: {
          category,
          target_goal: targetGoal,
          daily_budget: parseFloat(dailyBudget),
        },
        generatedCopy: activeCopy,
        timestamp: new Date().toISOString(),
      };

      const response = await fetch(webhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`n8n respondió con error: ${response.statusText}`);
      }

      setSuccessMsg('✓ Payload enviado al webhook de n8n con éxito. Workflow disparado.');
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Fallo de red. Verifica la URL del webhook de n8n y si está escuchando.');
    } finally {
      setLoadingSubmit(false);
    }
  };

  return (
    <div className="min-h-screen p-6 lg:p-12 flex flex-col items-center justify-center">
      <div className="w-full max-w-2xl bg-white dark:bg-gray-800 rounded-3xl shadow-xl border border-gray-150 dark:border-gray-700/60 p-6 lg:p-8 space-y-6">
        
        {/* Header */}
        <div>
          <h2 className="text-2xl font-bold text-gray-850 dark:text-white">Formulario y Despliegue de Campañas</h2>
          <p className="text-xs text-gray-400 mt-1">Configura parámetros y envía automatizaciones a n8n</p>
        </div>

        <form onSubmit={submitWorkflow} className="space-y-5">
          {/* Fila 1 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="block text-xs font-bold text-gray-400 dark:text-gray-400 uppercase tracking-wider">Categoría / Nicho</label>
              <input
                type="text"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                placeholder="Ej: E-commerce, Academia, SaaS"
                className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-750 border border-gray-200 dark:border-gray-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm dark:text-white transition-all"
              />
            </div>
            <div className="space-y-1">
              <label className="block text-xs font-bold text-gray-400 dark:text-gray-400 uppercase tracking-wider">Presupuesto Diario</label>
              <input
                type="number"
                value={dailyBudget}
                onChange={(e) => setDailyBudget(e.target.value)}
                placeholder="Monto diario"
                className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-750 border border-gray-200 dark:border-gray-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm dark:text-white transition-all"
              />
            </div>
          </div>

          {/* Objetivo */}
          <div className="space-y-1">
            <label className="block text-xs font-bold text-gray-400 dark:text-gray-400 uppercase tracking-wider">Objetivo de Conversión</label>
            <textarea
              value={targetGoal}
              onChange={(e) => setTargetGoal(e.target.value)}
              rows={3}
              placeholder="Describe qué deseas lograr con este anuncio..."
              className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-750 border border-gray-200 dark:border-gray-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm dark:text-white resize-none transition-all"
            />
          </div>

          {/* Webhook de n8n */}
          <div className="space-y-1">
            <label className="block text-xs font-bold text-gray-400 dark:text-gray-400 uppercase tracking-wider">Webhook de n8n (Test/Production URL)</label>
            <input
              type="url"
              value={webhookUrl}
              onChange={(e) => setWebhookUrl(e.target.value)}
              placeholder="https://servidor-n8n.co/webhook/..."
              className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-750 border border-gray-200 dark:border-gray-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm dark:text-white transition-all font-mono text-xs"
            />
          </div>

          {/* Acciones */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
            <button
              type="button"
              onClick={generateCopy}
              disabled={loadingCopy || loadingSubmit}
              className="bg-gray-800 dark:bg-gray-700 hover:bg-gray-900 text-white font-bold py-3 px-4 rounded-xl shadow-sm transition-all text-sm flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loadingCopy ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              <span>Crear Copy (IA)</span>
            </button>

            <button
              type="submit"
              disabled={loadingCopy || loadingSubmit}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-xl shadow-sm transition-all text-sm flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loadingSubmit ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              <span>Lanzar en n8n</span>
            </button>
          </div>
        </form>

        {/* Notificaciones */}
        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/50 rounded-2xl flex items-start gap-3 text-red-700 dark:text-red-400 text-xs">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
            <p className="font-semibold leading-relaxed">{error}</p>
          </div>
        )}

        {successMsg && (
          <div className="p-4 bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/50 rounded-2xl flex items-start gap-3 text-emerald-700 dark:text-emerald-400 text-xs">
            <CheckCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <p className="font-semibold leading-relaxed">{successMsg}</p>
          </div>
        )}

        {/* Preview del Copy de Anuncio */}
        {adCopy && (
          <div className="border border-gray-150 dark:border-gray-700/60 rounded-2xl p-4 bg-gray-50 dark:bg-gray-750/30 space-y-2.5 animate-fade-in">
            <span className="block text-[10px] text-gray-400 font-bold uppercase tracking-wider">Vista Previa Generada por LLM</span>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-gray-100 dark:border-gray-700/50 space-y-3">
              <h4 className="text-sm font-bold text-gray-900 dark:text-white">{adCopy.headline}</h4>
              <p className="text-xs text-gray-600 dark:text-gray-300 leading-relaxed font-sans">{adCopy.body}</p>
              {adCopy.cta && (
                <div className="flex justify-end pt-1 border-t border-gray-100 dark:border-gray-750">
                  <span className="text-[10px] bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 px-3 py-1.5 rounded font-bold uppercase tracking-wider">{adCopy.cta}</span>
                </div>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
