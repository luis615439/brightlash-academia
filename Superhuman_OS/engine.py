import json
import os
import sys

class SuperhumanEngine:
    def __init__(self, config_path="/Volumes/IA_LAB_DAT/INPUT_ZONE/AGENT_CONFIGS"):
        self.config_path = config_path
        self.agents = {}
        self.profile = {}
        self.load_all_configs()

    def load_all_configs(self):
        try:
            # Cargar perfiles maestros
            self.agents['alex'] = self._load_json("alex_supreme_v2.json")
            self.agents['merlin'] = self._load_json("merlin_integrated.json")
            self.agents['kaizen'] = self._load_json("kaizen_engine.json")
            self.agents['branding'] = self._load_json("branding_launch.json")
            
            # Cargar perfil humano (el filtro de verdad)
            exec_profile = self._load_json("execution_profile.json")
            self.profile = exec_profile.get("human_os_profile", {})
            self.coach = exec_profile.get("camali_coach", {})
            
            print("💎 Superhuman OS Engine: Motores lógicos cargados con éxito.")
        except Exception as e:
            print(f"❌ Error al cargar configuraciones: {e}")

    def _load_json(self, filename):
        full_path = os.path.join(self.config_path, filename)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def get_context(self, agent_name):
        """Genera el Mega-Prompt para el agente seleccionado"""
        agent = self.agents.get(agent_name, {})
        context = f"""
### MODO: {agent_name.upper()} SUPREME
IDENTIDAD: {json.dumps(agent, indent=2, ensure_ascii=False)}

### PERFIL DEL FUNDADOR (FILTRO DE VERDAD):
{json.dumps(self.profile, indent=2, ensure_ascii=False)}

### REGLAS DE ORO:
- No suavizar fallas.
- Priorizar impacto sobre complejidad.
- Una sola pregunta o acción a la vez.
"""
        return context

    def audit_idea(self, idea):
        """Prototipo de auditoría rápida usando lógica de Alex"""
        print(f"\n🔍 AUDITANDO IDEA: {idea}")
        print("-" * 30)
        # Aquí se integraría con el LLM enviando el contexto generado por get_context
        print("💡 [LOGICA ALEX]: Analizando cuello de botella...")
        print("💡 [LOGICA MERLIN]: Validando alineación de energía...")
        print("✅ Listo para consulta profunda.")

if __name__ == "__main__":
    engine = SuperhumanEngine()
    if len(sys.argv) > 1:
        engine.audit_idea(" ".join(sys.argv[1:]))
    else:
        print("\n🚀 USO: python3 engine.py 'Tu idea o problema aquí'")
