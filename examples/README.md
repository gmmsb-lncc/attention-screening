# 📚 Exemplos do DockTKinase

Esta pasta contém exemplos práticos demonstrando as funcionalidades do sistema DockTKinase.

## 🎯 Exemplos Disponíveis

### 🔧 Sistema de Configuração
**`exemplo_config_management.py`**
- Demonstra o sistema centralizado de configurações (Ponto 3)
- Templates predefinidos (development/production/research)
- Auto-configuração baseada em dados e recursos
- Validação robusta de configurações
- Serialização em múltiplos formatos (JSON/YAML/TOML)
- Integração completa com MLPPipeline

### 🖥️ Gerenciamento de Devices
**`exemplo_device_management.py`**
- Demonstra o sistema robusto de device management (Ponto 2)
- Detecção automática de CPU/CUDA/MPS
- Fallback inteligente GPU → CPU
- Benchmarking opcional para seleção
- Validação contínua de devices
- Integração com MLPPipeline

## 🚀 Como Executar

### Executar Exemplo Específico
```bash
cd /path/to/docktkinase

# Exemplo de configuração
python examples/exemplo_config_management.py

# Exemplo de device management
python examples/exemplo_device_management.py
```

### Saída Esperada

#### Config Management
```
🚀 SISTEMA DE CONFIGURAÇÃO CENTRALIZADA - PONTO 3
============================================================

📋 EXEMPLO 1: Templates Predefinidos
✅ Template development criado
✅ Template production criado
✅ Template research criado

🔧 EXEMPLO 2: Auto-Configuração
📊 Dataset Pequeno: Modelo [64, 32], Batch 32, LR 5e-04
📊 Dataset Grande: Modelo [256, 128, 64], Batch 32, LR 1e-03

💾 EXEMPLO 3: Serialização
✅ Config salvo em JSON/YAML/TOML
✅ Carregamento verificado
```

#### Device Management
```
🚀 SISTEMA DE DEVICE MANAGEMENT - PONTO 2
============================================================

🔍 Sistema detectado: 2 devices
1. Apple Metal Performance Shaders (mps) - ✅ RECOMENDADO
2. CPU (arm) | 16.0GB - 💡 disponível

🎯 RECOMENDAÇÃO FINAL: Apple Metal Performance Shaders (mps)
```

## 📋 Recursos Demonstrados

### Config Management
- ✅ Templates predefinidos com validação
- ✅ Auto-configuração inteligente
- ✅ Overrides com dot notation
- ✅ Serialização multiplataforma
- ✅ Validação robusta com errors/warnings

### Device Management  
- ✅ Detecção automática de hardware
- ✅ Benchmarking opcional
- ✅ Fallback inteligente
- ✅ Validação contínua
- ✅ Suporte MPS/CUDA/CPU

## 🛠️ Personalização

### Modificar Exemplos
1. Ajustar parâmetros nos exemplos
2. Testar diferentes configurações
3. Experimentar com seus próprios dados

### Criar Novos Exemplos
1. Seguir padrão `exemplo_nome_funcionalidade.py`
2. Incluir documentação detalhada
3. Demonstrar uso prático
4. Incluir tratamento de erros

## 🎓 Casos de Uso

### Para Desenvolvedores
- Entender APIs e funcionalidades
- Ver padrões de uso recomendados
- Testar configurações específicas

### Para Usuários Finais
- Aprender configuração do sistema
- Ver exemplos práticos
- Solucionar problemas comuns

## 🔗 Próximos Passos

Após executar os exemplos:
1. **Executar testes**: `python tests/validacao_final.py`
2. **Configurar sistema**: `python setup.py`  
3. **Ler documentação**: `GUIA_USUARIO.md`
4. **Explorar código**: `src/classifier/`
