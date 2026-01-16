# Conceitos Fundamentais: semantic-screening e DT-Kinase

## 📌 Definições

### semantic-screening
**Plataforma aberta e extensível** para predição de propriedades de interações proteína-ligante usando aprendizado profundo baseado em linguagem de proteína.

**Escopo**:
- Implementação completa de pipeline: embeddings → processamento → predição
- Suporte para múltiplas estratégias:
  - **Modelos clássicos**: 12 algoritmos de ML (classificação + regressão)
  - **Arquitetura neural**: DT-Kinase (CNN + Cross-Attention)
  - **Modelos de embeddings**: ESM-2, ESM-C, Boltz-2 (proteína), SMI-TED (ligante)
- Estratificação robusta com validação de leakage
- Modular e reutilizável para novos arquivos, modelos e abordagens

**Analogia**: semantic-screening é como um "toolkit" – fornece componentes reutilizáveis e padrões para construir soluções de screening.

---

### DT-Kinase
**Arquitetura neural específica** implementada na plataforma semantic-screening que soluciona o problema de predição de seletividade de quinases através de reformulação semântica.

**Componentes**:
1. **Codificação de Proteína**: Embeddings contextuais per-resíduo de modelos de linguagem de proteína (ESM-2, ESM-C, Boltz-2)
   - Capturam informação evolutiva implícita em sequência
   - Não requerem estrutura 3D experimental
   
2. **Codificação de Ligante**: Embeddings per-átomo de modelos de fundação química (SMI-TED)
   - Capturam propriedades moleculares e sintaxe SMILES
   - Codificam padrões de estrutura 2D/semântica
   
3. **Extração de Features Locais**: Codificadores CNN multi-escala
   - Kernels {3, 5, 7} para capturar padrões em múltiplas escalas
   - Conexões residuais preservam hierarquia de features
   
4. **Modelagem de Interação Semântica**: Mecanismos de Cross-Attention bidirecional
   - Proteína → Ligante: "Quais resíduos se ligam a quais átomos?"
   - Ligante → Proteína: "Quais átomos interagem com quais resíduos?"
   - Multi-Head (8 cabeças) para capturar diferentes tipos de interação
   
5. **Predição Multi-Tarefa**:
   - **Classificação**: Ativo/Inativo (logits binários)
   - **Regressão**: Valor de afinidade em escala pChEMBL (contínuo)
   - Otimização conjunta com ponderação por tarefa

**Analogia**: DT-Kinase é uma arquitetura específica – assim como "AlexNet" é uma arquitetura CNN específica dentro do ecossistema mais amplo de deep learning.

---

## 🔗 Relação Conceitual

```
┌────────────────────────────────────────────────────────────────┐
│                 SEMANTIC-SCREENING                             │
│        (Plataforma aberta de screening semântico)              │
│                                                                │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  ML CLASSICALS   │         │   DT-KINASE      │             │
│  │  12 algorithms   │         │   (CNN + Cross-  │             │
│  │  (RF, XGB, etc.) │         │    Attention)    │             │
│  └──────────────────┘         └──────────────────┘             │
│                                                                │
│  ┌──────────────────────────────────────────────────┐          │
│  │  EMBEDDING INFRASTRUCTURE                        │          │
│  │  • ESM-2 / ESM-C / Boltz-2 (Protein)            │          │
│  │  • SMI-TED (Ligand)                              │          │
│  │  • Cached embeddings & validation                │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                │
│  ┌──────────────────────────────────────────────────┐          │
│  │  STRATIFICATION & VALIDATION                     │          │
│  │  • Agglomerative clustering                      │          │
│  │  • Cosine similarity validation                  │          │
│  │  • Train/Val/Test splitting                      │          │
│  └──────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Quando Usar Cada Um

### semantic-screening
Use quando você quer:
- Uma **plataforma modular completa** para screening de interações proteína-ligante
- **Explorar múltiplas abordagens**: comparar ML clássico vs deep learning
- **Customização**: adicionar novos modelos, embeddings ou estratégias
- **Produção**: escalabilidade e robustez validadas
- **Investigação**: entender quais componentes afetam desempenho

### DT-Kinase (dentro de semantic-screening)
Use quando você quer:
- **Aproveitar a informação semântica** de proteínas e moléculas via PLMs
- **Modelar interações explicáveis** com mecanismos de atenção
- **Performance otimizada**: CNN captura local, Cross-Attention captura interações
- **Sem estrutura 3D**: aplicável a qualquer proteína com sequência conhecida
- **Multi-tarefa**: classificação e regressão simultâneas com incerteza

---

## 📊 Casos de Uso

### Exemplo 1: Descoberta de Inibidores de Quinase Bacteriana
**Cenário**: Você tem dataset de 15K moléculas contra 42 quinases bacterianas com dados de afinidade.

**Abordagem semantic-screening + DT-Kinase**:
1. Gerar embeddings ESM-C para 42 quinases (uma vez)
2. Gerar embeddings SMI-TED para 15K moléculas
3. Treinar DT-Kinase para predição multi-tarefa
4. Usar modelos ML clássicos como baselines
5. Comparar: DT-Kinase vs 12 algoritmos ML
6. Estratificar robustamente com validação de leakage

**Resultado**: Arquitetura neural especializada para seu problema + validação contra múltiplas baselines.

---

### Exemplo 2: Screening de Biblioteca Química Ultra-Grande
**Cenário**: Você tem 1B de moléculas e quer predizer atividade contra 100 proteínas.

**Abordagem semantic-screening + DT-Kinase**:
1. Treinar DT-Kinase uma vez em dataset benchmark
2. Gerar embeddings para 100 proteínas (cache reutilizável)
3. Processar 1B moléculas em batches (forward pass puro = fast)
4. Ranking de candidatos por score de afinidade predita

**Resultado**: Triagem de bilhões de compostos em horas, sem estrutura 3D, com uncertainties.

---

### Exemplo 3: Novo Alvo Sem Estrutura Cristalográfica
**Cenário**: Novo target com sequência anotada mas sem estrutura PDB.

**Abordagem semantic-screening + DT-Kinase**:
1. PLM (ESM-2) reconstruiu estrutura local implicitamente em embeddings
2. DT-Kinase não precisa de 3D explícito
3. Aplicável imediatamente a targets "orphaned"

**Resultado**: Screening funcional sem cristalografia.

---

## 🔬 Fundamentação Teórica

### Por que semantic-screening?
- **Sequência contém informação estrutural**: Demonstrado por AlphaFold2, ESMFold, ProtBERT
- **PLMs aprendem semântica**: Via auto-atenção em centenas de milhões de sequências
- **Reformulação do problema**: Não "Qual é a geometria?" mas "Quão compatível é a semântica?"
- **Universalidade**: Aplicável a qualquer proteína com sequência, estrutura ou não

### Por que DT-Kinase como arquitetura específica?
- **CNN**: Captura padrões locais em sequências e moléculas
- **Cross-Attention**: Modela compatibilidade semântica entre proteína e ligante
- **Multi-Tarefa**: Classificação + Regressão com ponderação de tarefa
- **Escalonável**: Forward pass puro, sem bottleneck geométrico

---

## 📚 Referências na Tese

- **Capítulo 1, Seção 1.3**: "From Docking to Language Modeling: The DockTKinase Philosophy"
  - Explica por que abandonar representações 3D
  - Fundamenta uso de PLMs

- **Capítulo 1, Seção 1.5**: "DT-Kinase: Objetivos e Contribuições"
  - Define DT-Kinase como arquitetura proposta
  - Especifica componentes: embeddings + CNN + Cross-Attention

- **Capítulo 2**: Estado da arte
  - Limitações de docking
  - Limitações de painéis experimentais
  - Necessidade de abordagem semântica

- **Capítulo 3**: Fundamentos teóricos
  - Formação matemática de PLMs
  - Atenção cruzada
  - Arquitetura proposta

---

## ✅ Checklist Conceitual

- [ ] semantic-screening = Plataforma aberta e modular
- [ ] DT-Kinase = Arquitetura neural específica dentro de semantic-screening
- [ ] semantic-screening implementa múltiplas abordagens (ML clássico + DL)
- [ ] DT-Kinase é otimizado para quinases e interações proteína-ligante
- [ ] Ambos operam sem requerer estrutura 3D
- [ ] semantic-screening fornece infra de embeddings, validação, estratificação
- [ ] DT-Kinase fornece arquitetura CNN + Cross-Attention para interações

---

**Última atualização**: Janeiro 2026  
**Documento**: Esclarecimento conceitual  
**Status**: Referência para toda documentação
