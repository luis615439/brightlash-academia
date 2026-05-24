import React, { useState, useEffect } from 'react'
import './index.css'

const API_URL = 'http://localhost:8001'

const SuperhumanPortal = () => {
  const [activeTab, setActiveTab] = useState('agents')
  const [selectedAgent, setSelectedAgent] = useState<any>(null)
  const [profileData, setProfileData] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  // Audit States
  const [auditQuery, setAuditQuery] = useState('')
  const [auditAgent, setAuditAgent] = useState('alex')
  const [auditResult, setAuditResult] = useState<string | null>(null)
  const [auditLoading, setAuditLoading] = useState(false)

  const getAgentInfo = (data: any) => {
    if (!data) return { name: 'Sincronizando...', mission: 'Esperando directivas del volumen externo.', rules: [] }
    
    // Extracción inteligente de datos (Deep search)
    const core = data.alex_os || data.kaizen_engine || data.identity || data.merlin_integrated || data
    
    return {
      name: core.name || core.identity || core.role || 'PROTOCOLO ACTIVO',
      mission: core.mission || core.core_function || core.purpose || 'Misión estratégica cargada en el núcleo.',
      rules: core.core_rules || core.core_principles || core.rules || core.kaizen_loop?.map((l: any) => l.description) || []
    }
  }

  const loadAgentDetails = async (id: string) => {
    setLoading(true)
    setSelectedAgent(null) // Reset para mostrar carga
    try {
      const response = await fetch(`${API_URL}/api/agents/${id}`)
      if (!response.ok) throw new Error('Error al cargar agente')
      const data = await response.json()
      console.log('Data received:', data)
      setSelectedAgent(data)
    } catch (error) {
      console.error('Error loading agent:', error)
      alert('Error de conexión. Asegúrate de que el backend de FastAPI esté corriendo en el puerto 8001.')
    } finally {
      setLoading(false)
    }
  }

  const loadProfile = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/api/agents/profile`)
      if (!response.ok) throw new Error('Error al cargar perfil')
      const data = await response.json()
      setProfileData(data)
    } catch (error) {
      console.error('Error loading profile:', error)
    } finally {
      setLoading(false)
    }
  }

  const runAudit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!auditQuery.trim()) return
    
    setAuditLoading(true)
    setAuditResult(null)
    try {
      const response = await fetch(`${API_URL}/api/audit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          idea: auditQuery,
          agent_id: auditAgent
        })
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Error en el servidor de auditoría')
      }
      
      const data = await response.json()
      setAuditResult(data.audit)
    } catch (error: any) {
      console.error('Error during audit:', error)
      setAuditResult(`### ❌ ERROR DE CONEXIÓN\nNo se pudo completar la auditoría.\n\n**Detalle:** ${error.message || 'El servidor FastAPI en la red local no está activo o la API Key no está cargada.'}`)
    } finally {
      setAuditLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'profile') loadProfile()
  }, [activeTab])

  const agentInfo = getAgentInfo(selectedAgent)

  // Custom Markdown parser for terminal output
  const renderAuditResult = (text: string) => {
    return text.split('\n').map((line, index) => {
      // Headers
      if (line.startsWith('### ')) {
        return <h3 key={index} style={{ color: 'var(--alex-color)', marginTop: '1.5rem', marginBottom: '0.8rem', letterSpacing: '1px', textTransform: 'uppercase', borderBottom: '1px solid rgba(255,0,85,0.2)', paddingBottom: '0.3rem' }}>{line.replace('### ', '')}</h3>
      }
      if (line.startsWith('## ')) {
        return <h2 key={index} style={{ color: 'var(--kaizen-color)', marginTop: '2rem', marginBottom: '1rem', letterSpacing: '2px' }}>{line.replace('## ', '')}</h2>
      }
      if (line.startsWith('# ')) {
        return <h1 key={index} style={{ color: 'var(--brando-color)', marginTop: '2rem', marginBottom: '1rem' }}>{line.replace('# ', '')}</h1>
      }
      
      // List items
      if (line.startsWith('- ') || line.startsWith('* ')) {
        const content = line.substring(2)
        return (
          <li key={index} style={{ marginLeft: '1.5rem', marginBottom: '0.6rem', listStyleType: 'square', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
            {parseBoldText(content)}
          </li>
        )
      }

      // Normal paragraph
      if (line.trim() === '') return <div key={index} style={{ height: '0.8rem' }} />

      return (
        <p key={index} style={{ marginBottom: '0.8rem', lineHeight: '1.6', color: '#fff' }}>
          {parseBoldText(line)}
        </p>
      )
    })
  }

  const parseBoldText = (text: string) => {
    const parts = text.split(/\*\*([\s\S]*?)\*\*/)
    return parts.map((part, i) => {
      if (i % 2 === 1) {
        return <strong key={i} style={{ color: 'var(--kaizen-color)', fontWeight: 'bold' }}>{part}</strong>
      }
      return part
    })
  }

  return (
    <div className="content-layer">
      <div className="hud-container"></div>
      
      <div className="main-wrapper" style={{ padding: '2rem' }}>
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <div>
            <h1 className="title-glow" style={{ margin: 0 }}>Superhuman OS <span style={{color: 'var(--kaizen-color)'}}>Engine</span></h1>
            <p style={{fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0.3rem 0 0 0'}}>CENTRO DE AUDITORIA EMPRESARIAL</p>
          </div>
          <div className="tag-system" style={{ borderColor: (loading || auditLoading) ? 'orange' : 'var(--kaizen-color)', color: (loading || auditLoading) ? 'orange' : 'var(--kaizen-color)' }}>
            STATUS: {(loading || auditLoading) ? 'SYNCING DATA...' : 'OPERATIONAL'}
          </div>
        </header>

        <nav style={{ display: 'flex', gap: '2rem', marginBottom: '2rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '1rem' }}>
          {['AGENTS', 'DIAGNOSTIC', 'PROFILE', 'KAIZEN'].map(tab => (
            <button 
              key={tab}
              onClick={() => { setActiveTab(tab.toLowerCase()); setSelectedAgent(null); }}
              style={{
                background: 'none',
                border: 'none',
                color: activeTab === tab.toLowerCase() ? 'var(--kaizen-color)' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontWeight: 'bold',
                letterSpacing: '2px',
                fontSize: '0.9rem',
                transition: 'all 0.3s'
              }}
            >
              {tab}
            </button>
          ))}
        </nav>

        <main>
          {activeTab === 'agents' && !selectedAgent && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '2rem' }}>
              <AgentCard name="ALEX SUPREME" role="Estratega / ROI" color="var(--alex-color)" mission="Detectar el bottleneck real y generar dinero." onAction={() => loadAgentDetails('alex')} />
              <AgentCard name="MERLIN" role="Energía / Foco" color="var(--merlin-color)" mission="Alinear el estado interno con la acción correcta." onAction={() => loadAgentDetails('merlin')} />
              <AgentCard name="BRANDO" role="Branding / Inevitable" color="var(--brando-color)" mission="Eliminar lo genérico y crear marcas inevitables." onAction={() => loadAgentDetails('branding')} />
              <AgentCard name="KAIZEN V1" role="Fundamentos" color="var(--kaizen-color)" mission="Construir las bases operativas sólidas del negocio." onAction={() => loadAgentDetails('kaizen_v1')} />
              <AgentCard name="KAIZEN V2" role="Escalamiento" color="var(--kaizen-color)" mission="Ciclos avanzados de optimización basados en métricas." onAction={() => loadAgentDetails('kaizen_v2')} />
            </div>
          )}

          {selectedAgent && (
            <div className="agent-detail-view" style={{ animation: 'fadeIn 0.5s ease', border: '1px solid var(--glass-border)', padding: '2rem', background: 'var(--card-bg)', borderRadius: '8px' }}>
              <button onClick={() => setSelectedAgent(null)} style={{ color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer', marginBottom: '1rem', fontSize: '0.8rem' }}>← VOLVER AL HUB CENTRAL</button>
              <h2 style={{ color: 'var(--kaizen-color)', marginBottom: '1.5rem', textTransform: 'uppercase', letterSpacing: '3px' }}>{agentInfo.name}</h2>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '3rem' }}>
                <div style={{ borderRight: '1px solid var(--glass-border)', paddingRight: '2rem' }}>
                  <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '1rem', letterSpacing: '1px' }}>Directiva Principal</h4>
                  <p style={{ lineHeight: '1.8', color: '#fff', fontSize: '1.1rem', fontStyle: 'italic' }}>"{agentInfo.mission}"</p>
                  
                  <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '4px' }}>
                    <h5 style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>LOGICA DE PROCESAMIENTO</h5>
                    <p style={{ fontSize: '0.8rem' }}>Agente operando bajo el volumen IA_LAB_DAT. Estado: Persistente.</p>
                  </div>
                </div>
                <div>
                  <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '1.5rem', letterSpacing: '1px' }}>Protocolos de Oro</h4>
                  <ul style={{ listStyle: 'none', fontSize: '0.95rem' }}>
                    {agentInfo.rules.length > 0 ? agentInfo.rules.map((rule: string, i: number) => (
                      <li key={i} style={{ marginBottom: '1.2rem', color: 'var(--kaizen-color)', borderLeft: '3px solid', paddingLeft: '1rem', lineHeight: '1.4' }}>{rule}</li>
                    )) : <li style={{ color: 'var(--text-secondary)' }}>Sincronizando reglas con el volumen externo...</li>}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'profile' && (
            <div style={{ padding: '2rem', background: 'var(--card-bg)', border: '1px solid var(--brando-color)', borderRadius: '8px' }}>
              <h2 style={{ color: 'var(--brando-color)', marginBottom: '1.5rem', letterSpacing: '2px' }}>HUMAN OPERATING SYSTEM (PROFILE)</h2>
              {profileData ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.5rem', borderRadius: '4px', border: '1px solid rgba(201, 169, 110, 0.1)' }}>
                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', marginBottom: '1rem' }}>REGLAS DIARIAS</h4>
                    <pre style={{ fontSize: '0.85rem', color: 'var(--brando-color)', whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                      {JSON.stringify(profileData.execution_profile?.daily_rules || profileData.daily_rules, null, 2)}
                    </pre>
                  </div>
                  <div>
                    <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', marginBottom: '1rem' }}>NO NEGOCIABLES</h4>
                    <ul style={{ listStyle: 'none' }}>
                      {(profileData.execution_profile?.non_negotiables || profileData.non_negotiables || []).map((n: string, i: number) => (
                        <li key={i} style={{ marginBottom: '1rem', color: '#fff', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>{n}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              ) : <div className="loading-shimmer">CONECTANDO CON EL PERFIL EN IA_LAB_DAT...</div>}
            </div>
          )}

          {activeTab === 'kaizen' && (
            <div style={{ padding: '2rem', background: 'var(--card-bg)', border: '1px solid var(--kaizen-color)', borderRadius: '8px' }}>
              <h2 style={{ color: 'var(--kaizen-color)', marginBottom: '2rem', letterSpacing: '2px' }}>MOTOR KAIZEN (ANALITICA)</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
                <MetricBox title="CONVERSION RATE" value="8.4%" color="var(--kaizen-color)" />
                <MetricBox title="ENERGY STABILITY" value="92%" color="var(--merlin-color)" />
                <MetricBox title="REVENUE / FRICTION" value="OPTIMAL" color="var(--alex-color)" />
              </div>
              <div style={{ marginTop: '3rem', padding: '1.5rem', background: 'rgba(0,255,209,0.03)', borderRadius: '4px', border: '1px dashed var(--kaizen-color)' }}>
                <h4 style={{ fontSize: '0.8rem', color: 'var(--kaizen-color)', marginBottom: '1rem' }}>Siguiente Salto (Kaizen Next):</h4>
                <p style={{ fontSize: '0.9rem' }}>Optimización del guión de cierre en WhatsApp basado en el segmento 1A (Principiantes).</p>
              </div>
            </div>
          )}

          {activeTab === 'diagnostic' && (
            <div style={{ padding: '2rem', border: '1px solid var(--alex-color)', background: 'rgba(255, 0, 85, 0.02)', borderRadius: '8px' }}>
              <h2 style={{color: 'var(--alex-color)', marginBottom: '1.5rem', letterSpacing: '2px'}}>MODO ZUCK: AUDITORIA BRUTAL</h2>
              
              <form onSubmit={runAudit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div>
                  <label style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', display: 'block', marginBottom: '0.8rem' }}>
                    Seleccionar Mente de Auditoría
                  </label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
                    {[
                      { id: 'alex', label: 'Alex Supreme', desc: 'Estrategia / ROI', color: 'var(--alex-color)' },
                      { id: 'merlin', label: 'Merlin', desc: 'Alineación / Energía', color: 'var(--merlin-color)' },
                      { id: 'branding', label: 'Brando', desc: 'Marketing / Identidad', color: 'var(--brando-color)' },
                      { id: 'kaizen_v1', label: 'Kaizen V1', desc: 'Bases y Fundamentos', color: 'var(--kaizen-color)' },
                      { id: 'kaizen_v2', label: 'Kaizen V2', desc: 'Métricas y Crecimiento', color: 'var(--kaizen-color)' }
                    ].map(agent => (
                      <button
                        key={agent.id}
                        type="button"
                        onClick={() => setAuditAgent(agent.id)}
                        style={{
                          background: auditAgent === agent.id ? `${agent.color}15` : 'rgba(0,0,0,0.5)',
                          border: `1px solid ${auditAgent === agent.id ? agent.color : 'var(--glass-border)'}`,
                          color: auditAgent === agent.id ? '#fff' : 'var(--text-secondary)',
                          padding: '0.8rem 1.5rem',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          textAlign: 'left',
                          flex: '1 1 200px',
                          transition: 'all 0.3s'
                        }}
                      >
                        <div style={{ fontWeight: 'bold', fontSize: '0.9rem', color: auditAgent === agent.id ? agent.color : 'var(--text-secondary)' }}>{agent.label}</div>
                        <div style={{ fontSize: '0.7rem', marginTop: '0.2rem' }}>{agent.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label htmlFor="audit-query" style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', display: 'block', marginBottom: '0.5rem' }}>
                    Ingresa tu idea de negocio o campaña
                  </label>
                  <textarea
                    id="audit-query"
                    value={auditQuery}
                    onChange={(e) => setAuditQuery(e.target.value)}
                    style={{
                      width: '100%',
                      height: '150px',
                      background: '#000',
                      border: '1px solid #333',
                      color: '#fff',
                      padding: '1.5rem',
                      borderRadius: '4px',
                      outline: 'none',
                      fontSize: '1rem',
                      lineHeight: '1.5',
                      fontFamily: 'monospace'
                    }}
                    placeholder="Ejemplo: Quiero lanzar una suscripción mensual de café premium orgánico directo de Oaxaca para programadores y diseñadores en CDMX..."
                    disabled={auditLoading}
                  />
                </div>

                <button
                  type="submit"
                  disabled={auditLoading || !auditQuery.trim()}
                  style={{
                    alignSelf: 'flex-start',
                    background: 'var(--alex-color)',
                    color: '#fff',
                    border: 'none',
                    padding: '1rem 3rem',
                    cursor: 'pointer',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    letterSpacing: '2px',
                    opacity: (auditLoading || !auditQuery.trim()) ? 0.5 : 1,
                    transition: 'all 0.3s'
                  }}
                >
                  {auditLoading ? 'ANALIZANDO DESTRUCTIVAMENTE...' : 'DESTRUIR BULLSHIT'}
                </button>
              </form>

              {(auditLoading || auditResult) && (
                <div style={{
                  marginTop: '2rem',
                  border: '1px solid var(--glass-border)',
                  background: '#050505',
                  borderRadius: '4px',
                  padding: '2rem',
                  fontFamily: 'monospace',
                  minHeight: '200px',
                  boxShadow: 'inset 0 0 20px rgba(0,0,0,0.8)'
                }}>
                  {auditLoading ? (
                    <div className="loading-shimmer" style={{ color: 'var(--alex-color)' }}>
                      [CONECTANDO NÚCLEO COGNITIVO... EJECUTANDO PROTOCOLO DE AUDITORÍA FORENSE...]
                    </div>
                  ) : (
                    <div style={{ color: '#fff', fontSize: '0.95rem' }}>
                      {renderAuditResult(auditResult || '')}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

const AgentCard = ({ name, role, color, mission, onAction }: any) => (
  <div style={{ padding: '1.5rem', background: 'var(--card-bg)', border: `1px solid ${color}44`, borderRadius: '8px', position: 'relative', transition: 'transform 0.3s' }}>
    <div style={{ position: 'absolute', top: 0, left: 0, width: '4px', height: '100%', background: color }}></div>
    <h3 style={{ color: color, fontSize: '1.3rem', letterSpacing: '1px', margin: '0 0 0.5rem 0' }}>{name}</h3>
    <p style={{ fontSize: '0.75rem', color: color, marginBottom: '1.5rem', textTransform: 'uppercase', margin: '0 0 1rem 0' }}>{role}</p>
    <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: '1.4', margin: 0 }}>{mission}</p>
    <button onClick={onAction} style={{ marginTop: '2rem', background: color, color: '#000', border: 'none', padding: '0.6rem 1.2rem', fontSize: '0.75rem', cursor: 'pointer', fontWeight: 'bold', width: '100%', textTransform: 'uppercase', letterSpacing: '1px' }}>ACTIVAR PROTOCOLO</button>
  </div>
)

const MetricBox = ({ title, value, color }: any) => (
  <div style={{ border: `1px solid ${color}33`, padding: '1.5rem', borderRadius: '8px', background: 'rgba(255,255,255,0.01)', position: 'relative' }}>
    <h5 style={{ color: 'var(--text-secondary)', fontSize: '0.65rem', marginBottom: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px', margin: 0 }}>{title}</h5>
    <div style={{ color: color, fontSize: '1.8rem', fontWeight: '800', textShadow: `0 0 10px ${color}33`, marginTop: '0.5rem' }}>{value}</div>
  </div>
)

export default SuperhumanPortal
