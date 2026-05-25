import React, { useState } from 'react';
import { DollarSign, ShieldAlert, TrendingUp, AlertTriangle, CheckCircle, RefreshCw } from 'lucide-react';

interface AuditResult {
  status: 'VERDE' | 'AMARILLO' | 'ROJO';
  cost_per_conversion: number;
  limit_threshold: number;
  alert_message: string;
  action: 'PAUSAR' | 'OPTIMIZAR' | 'ESCALAR';
}

export default function GenericMetricsAuditor() {
  const [totalSpend, setTotalSpend] = useState<string>('');
  const [totalConversions, setTotalConversions] = useState<string>('');
  const [unitValue, setUnitValue] = useState<string>('');
  const [thresholdPercent, setThresholdPercent] = useState<string>('75');
  const [apiKey, setApiKey] = useState<string>('TU_API_KEY_SECRETA_AQUI');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AuditResult | null>(null);

  const handleAudit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!totalSpend || !totalConversions || !unitValue) {
      setError('Por favor, completa todos los campos requeridos.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/audit-metrics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
        },
        body: JSON.stringify({
          total_spend: parseFloat(totalSpend),
          total_conversions: parseInt(totalConversions, 10),
          unit_value: parseFloat(unitValue),
          threshold_percent: parseFloat(thresholdPercent),
        }),
      });

      if (!response.ok) {
        if (response.status === 403) {
          throw new Error('API Key incorrecta o acceso denegado.');
        }
        throw new Error('Error al procesar la auditoría en el servidor.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'No se pudo conectar con el servidor. Asegúrate de que el backend de FastAPI esté activo en el puerto 8000.');
    } finally {
      setLoading(false);
    }
  };

  const getStatusStyles = (status: 'VERDE' | 'AMARILLO' | 'ROJO') => {
    switch (status) {
      case 'ROJO':
        return {
          bg: 'bg-red-50 dark:bg-red-950/20 border-red-200 dark:border-red-900/50',
          text: 'text-red-700 dark:text-red-400',
          lightColor: 'bg-red-500 shadow-red-500/50',
          icon: ShieldAlert,
          titleColor: 'text-red-800 dark:text-red-300',
        };
      case 'AMARILLO':
        return {
          bg: 'bg-amber-50 dark:bg-amber-950/20 border-amber-200 dark:border-amber-900/50',
          text: 'text-amber-700 dark:text-amber-400',
          lightColor: 'bg-amber-500 shadow-amber-500/50',
          icon: AlertTriangle,
          titleColor: 'text-amber-800 dark:text-amber-300',
        };
      case 'VERDE':
        return {
          bg: 'bg-emerald-50 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-900/50',
          text: 'text-emerald-700 dark:text-emerald-400',
          lightColor: 'bg-emerald-500 shadow-emerald-500/50',
          icon: CheckCircle,
          titleColor: 'text-emerald-800 dark:text-emerald-300',
        };
    }
  };

  const statusStyle = result ? getStatusStyles(result.status) : null;
  const StatusIcon = statusStyle ? statusStyle.icon : null;

  return (
    <div className="min-h-screen p-6 lg:p-12 flex flex-col items-center justify-center">
      <div className="w-full max-w-2xl bg-white dark:bg-gray-800 rounded-3xl shadow-xl border border-gray-150 dark:border-gray-700/60 p-6 lg:p-8 space-y-6">
        
        {/* Header */}
        <div className="flex items-center gap-4 border-b border-gray-150 dark:border-gray-700/50 pb-5">
          <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl text-white shadow-md">
            <DollarSign className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-850 dark:text-white">Auditor Financiero y de Métricas</h1>
            <p className="text-xs text-gray-400 mt-0.5">Control inteligente de margen y costos de conversión</p>
          </div>
        </div>

        {/* Formulario */}
        <form onSubmit={handleAudit} className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="space-y-1">
            <label className="block text-xs font-bold text-gray-400 dark:text-gray-400 uppercase tracking-wider">Inversión / Gasto Publicitario</label>
            <input
              type="number"
              step="any"
              value={totalSpend}
              onChange={(e) => setTotalSpend(e.target.value)}
              placeholder="Inversión total"
              className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-750 border border-gray-200 dark:border-gray-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm dark:text-white transition-all"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-bold text-gray-400 dark:text-gray-400 uppercase tracking-wider">Conversiones / Leads Totales</label>
            <input
              type="number"
              value={totalConversions}
              onChange={(e) => setTotalConversions(e.target.value)}
              placeholder="Ej: 15"
              className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-750 border border-gray-200 dark:border-gray-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm dark:text-white transition-all"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-bold text-gray-400 dark:text-gray-400 uppercase tracking-wider">Valor del Producto / Ticket</label>
            <input
              type="number"
              step="any"
              value={unitValue}
              onChange={(e) => setUnitValue(e.target.value)}
              placeholder="Precio del producto"
              className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-750 border border-gray-200 dark:border-gray-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm dark:text-white transition-all"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-bold text-gray-400 dark:text-gray-400 uppercase tracking-wider">Límite de Alerta (Porcentaje CPL)</label>
            <input
              type="number"
              value={thresholdPercent}
              onChange={(e) => setThresholdPercent(e.target.value)}
              placeholder="Porcentaje de corte. Ej: 75"
              className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-750 border border-gray-200 dark:border-gray-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm dark:text-white transition-all"
            />
          </div>

          <div className="md:col-span-2 space-y-1">
            <label className="block text-xs font-bold text-gray-400 dark:text-gray-400 uppercase tracking-wider">API Key del Sistema</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="X-API-KEY"
              className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-750 border border-gray-200 dark:border-gray-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm dark:text-white transition-all"
            />
          </div>

          <div className="md:col-span-2 pt-2">
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-3 px-4 rounded-xl shadow-md transition-all text-sm flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Analizando Métricas...</span>
                </>
              ) : (
                <>
                  <TrendingUp className="w-4 h-4" />
                  <span>Ejecutar Auditoría</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Panel de Error */}
        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/50 rounded-2xl flex items-start gap-3 text-red-700 dark:text-red-400 text-xs">
            <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5" />
            <p className="font-semibold leading-relaxed">{error}</p>
          </div>
        )}

        {/* Semáforo de Resultados */}
        {result && statusStyle && StatusIcon && (
          <div className={`p-6 border rounded-2xl ${statusStyle.bg} transition-all duration-500 animate-fade-in space-y-4`}>
            
            {/* Cabecera */}
            <div className="flex items-center justify-between border-b border-gray-200/50 dark:border-gray-750/30 pb-3">
              <div className="flex items-center gap-2.5">
                <span className={`w-3.5 h-3.5 rounded-full ${statusStyle.lightColor} animate-pulse`} />
                <h3 className={`font-extrabold text-sm uppercase tracking-wider ${statusStyle.titleColor}`}>
                  Estatus Financiero: {result.status}
                </h3>
              </div>
              <span className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1 bg-white/70 dark:bg-black/20 rounded-full ${statusStyle.text}`}>
                Acción: {result.action}
              </span>
            </div>

            {/* Métricas */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-0.5">
                <span className="block text-[10px] text-gray-400 uppercase">Costo por Conversión (CPL)</span>
                <span className={`text-2xl font-black ${statusStyle.text}`}>
                  ${result.cost_per_conversion.toLocaleString('es-MX', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} MXN
                </span>
              </div>
              <div className="space-y-0.5">
                <span className="block text-[10px] text-gray-400 uppercase">Límite de Alerta ({thresholdPercent}%)</span>
                <span className="text-xl font-bold text-gray-700 dark:text-gray-300">
                  ${result.limit_threshold.toLocaleString('es-MX', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} MXN
                </span>
              </div>
            </div>

            {/* Directiva */}
            <div className="flex items-start gap-3 bg-white/40 dark:bg-black/10 p-3.5 rounded-xl border border-white/60 dark:border-black/5">
              <StatusIcon className={`w-6 h-6 shrink-0 ${statusStyle.text}`} />
              <p className={`text-xs font-semibold leading-relaxed ${statusStyle.text}`}>
                {result.alert_message}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
