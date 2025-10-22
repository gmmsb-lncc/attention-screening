# 📚 Documentação DockTKinase

Índice completo da documentação do projeto DockTKinase.

---

## 🚀 Início Rápido

- **[QUICK_START.md](QUICK_START.md)** - Guia de início rápido
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Guia de instalação completo
- **[USER_GUIDE.md](USER_GUIDE.md)** - Manual do usuário
- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Guia de execução

---

## 🧬 Integração ESM-2 (Embeddings de Proteínas)

### Documentação Principal
- **[ESM_INTEGRATION.md](ESM_INTEGRATION.md)** ⭐ - Guia completo de integração ESM-2
  - Estrutura de arquivos
  - 6 modelos disponíveis (t6, t12, t30, t33, **t36** padrão, t48)
  - Configuração e uso
  - Troubleshooting e benchmarks

### Validação e Testes
- **[ESM_VALIDATION_REPORT.md](ESM_VALIDATION_REPORT.md)** - Relatório de validação completo
  - 8 testes realizados (100% aprovados)
  - Resultados de embedding real
  - Performance metrics
  - Requisitos de sistema

### Organização
- **[ORGANIZACAO_ARQUIVOS_ESM.md](ORGANIZACAO_ARQUIVOS_ESM.md)** - Estrutura de arquivos ESM
  - Localização de testes (`tests/`)
  - Localização de docs (`docs/`)
  - Como executar testes

### Histórico
- **[CHANGELOG_ESM.md](CHANGELOG_ESM.md)** - Registro de mudanças ESM-2
  - Commits realizados
  - Arquivos modificados
  - Impacto em dependências

---

## 🏗️ Arquitetura e Build

- **[PROJETO_FINAL_ESTRUTURA.md](PROJETO_FINAL_ESTRUTURA.md)** - Estrutura do projeto
- **[ESTRATEGIA_MODULARIZACAO_BUILD.md](ESTRATEGIA_MODULARIZACAO_BUILD.md)** - Estratégia de modularização
- **[RELATORIO_FINAL_MODULARIZACAO.md](RELATORIO_FINAL_MODULARIZACAO.md)** - Relatório de modularização
- **[RELATORIO_VALIDACAO_BUILD.md](RELATORIO_VALIDACAO_BUILD.md)** - Validação do build

---

## 📦 Configuração e Setup

- **[SETUP_SUMMARY.md](SETUP_SUMMARY.md)** - Resumo de configuração
- **[DEPENDENCY_RESOLUTION.md](DEPENDENCY_RESOLUTION.md)** - Resolução de dependências
- **[PATH_PORTABILITY_FIX.md](PATH_PORTABILITY_FIX.md)** - Correção de portabilidade

---

## ✅ Validação e Otimização

- **[OPTIMIZATION_VALIDATION.md](OPTIMIZATION_VALIDATION.md)** - Validação de otimizações
- **[PIPELINE_SUCCESS_REPORT.md](PIPELINE_SUCCESS_REPORT.md)** - Relatório de sucesso do pipeline
- **[RESOLUCAO_WARNINGS.md](RESOLUCAO_WARNINGS.md)** - Resolução de warnings

---

## 🧹 Manutenção e Limpeza

- **[CLEANUP_REPORT.md](CLEANUP_REPORT.md)** - Relatório de limpeza
- **[CLEANUP_KINASE_FOLDERS.md](CLEANUP_KINASE_FOLDERS.md)** - Limpeza de pastas kinase

---

## 🔧 Modularização (Build)

- **[STATUS_MODULARIZACAO.md](STATUS_MODULARIZACAO.md)** - Status da modularização

---

## ⚠️ Troubleshooting

- **[HUGGINGFACE_RATE_LIMIT.md](HUGGINGFACE_RATE_LIMIT.md)** - Solução de rate limit HuggingFace

---

## 📊 Estrutura de Documentação

```
docs/
│
├── 🚀 Início Rápido
│   ├── QUICK_START.md
│   ├── INSTALLATION_GUIDE.md
│   ├── USER_GUIDE.md
│   └── EXECUTION_GUIDE.md
│
├── 🧬 ESM-2 (Embeddings)
│   ├── ESM_INTEGRATION.md          ⭐ Principal
│   ├── ESM_VALIDATION_REPORT.md
│   ├── ORGANIZACAO_ARQUIVOS_ESM.md
│   └── CHANGELOG_ESM.md
│
├── 🏗️ Arquitetura
│   ├── PROJETO_FINAL_ESTRUTURA.md
│   ├── ESTRATEGIA_MODULARIZACAO_BUILD.md
│   ├── RELATORIO_FINAL_MODULARIZACAO.md
│   └── RELATORIO_VALIDACAO_BUILD.md
│
├── 📦 Setup
│   ├── SETUP_SUMMARY.md
│   ├── DEPENDENCY_RESOLUTION.md
│   └── PATH_PORTABILITY_FIX.md
│
└── ✅ Validação
    ├── OPTIMIZATION_VALIDATION.md
    ├── PIPELINE_SUCCESS_REPORT.md
    └── RESOLUCAO_WARNINGS.md
```

---

## 🎯 Documentos Mais Importantes

Para novos usuários, recomendamos ler nesta ordem:

1. **[QUICK_START.md](QUICK_START.md)** - Começar rapidamente
2. **[ESM_INTEGRATION.md](ESM_INTEGRATION.md)** - Entender embeddings ESM-2
3. **[USER_GUIDE.md](USER_GUIDE.md)** - Guia completo de uso
4. **[PROJETO_FINAL_ESTRUTURA.md](PROJETO_FINAL_ESTRUTURA.md)** - Entender estrutura

---

## 🔄 Atualizações Recentes

**22 de outubro de 2025**:
- ✅ Integração ESM-2 local completa
- ✅ Validação 100% (8/8 testes aprovados)
- ✅ Organização de arquivos ESM
- ✅ Documentação ESM completa (4 arquivos)

---

## 📝 Convenções

### Nomenclatura de Arquivos

- **`*_GUIDE.md`** - Guias práticos e tutoriais
- **`*_REPORT.md`** - Relatórios de validação e testes
- **`CHANGELOG_*.md`** - Registros de mudanças
- **`*_VALIDATION.md`** - Documentos de validação
- **`RELATORIO_*.md`** - Relatórios em português

### Ícones Usados

- ⭐ - Documento principal/importante
- ✅ - Validado/Aprovado
- 🚀 - Início rápido
- 🧬 - Relacionado a bioinformática
- 🏗️ - Arquitetura
- 📦 - Configuração
- 🔧 - Troubleshooting

---

## 💡 Contribuindo

Para adicionar nova documentação:

1. Criar arquivo `.md` em `docs/`
2. Seguir convenções de nomenclatura
3. Adicionar ao índice neste README
4. Incluir ícone apropriado
5. Adicionar à seção de atualizações recentes

---

**Última atualização**: 22 de outubro de 2025
