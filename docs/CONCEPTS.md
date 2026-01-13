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
**Arquitetura neural específica** implementada na plataforma semantic-screening que soluciona o paradoxo de seletividade de quinases através de reformulação semântica.

**O Paradoxo de Seletividade** (Capítulo 1, Seção 1.2 da tese):
- 518 quinases humanas compartilham arquitetura catalítica altamente conservada (RMSD < 2Å entre sítios ATP)
- Discriminação de 10-fold em seletividade requer diferença de ΔG de apenas 1.4 kcal/mol
- Erro sistemático de funções de scoring: ±2-3 kcal/mol (insuficiente para discriminar seletividade)

**Resolução Proposta** (Capítulo 3, Seção 3.8 da tese):
1. Embeddings contextuais integram informação de sequência completa (domínios regulatórios, loops de ativação, porções terminais)
2. Representação semântica codifica informação evolutiva implícita
3. Atenção cruzada aprende correspondências posição-específicas

**Componentes Arquiteturais** (Capítulo 3):
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

### Por que semantic-screening? (Capítulo 1, Seção 1.4)
- **Sequência contém informação estrutural**: O dogma central estabelece que sequência determina estrutura. AlphaFold2 e ESMFold demonstram que este mapeamento é computacionalmente recuperável—informação estrutural está *codificada* na sequência, não adicionada externamente
- **PLMs aprendem semântica evolutiva**: ESM-2 treinado em ~65M sequências via MLM aprende representações que capturam não apenas estrutura, mas padrões de co-evolução e restrições funcionais de pressão seletiva
- **Auto-atenção captura dependências globais**: Cada resíduo é representado em função de *toda* a sequência, capturando dependências de longo alcance inacessíveis a métodos que operam em representações locais
- **Reformulação semântica**: Substitui "Quão bem este ligante se encaixa neste sítio?" por "Quão compatíveis são as representações latentes desta proteína e deste ligante em espaço vetorial compartilhado?"
- **Universalidade**: Aplicável a qualquer proteína com sequência, incluindo ~40% das quinases sem estrutura experimental

### Por que DT-Kinase como arquitetura específica? (Capítulo 3)
- **CNN multi-escala (kernels 3,5,7)**: Captura padrões locais em sequências e moléculas em diferentes granularidades
- **Cross-Attention bidirecional**: Modela compatibilidade semântica entre proteína e ligante com interpretabilidade intrínseca
- **Multi-Tarefa**: Classificação + Regressão com loss combinada $\mathcal{L} = \alpha \cdot \mathcal{L}_{MSE} + (1-\alpha) \cdot \mathcal{L}_{BCE}$
- **Escalável**: Complexidade $\mathcal{O}(n^2 d + m^2 d + nmd)$ permite throughput >10⁶ pred/hora

---

## 📚 Referências na Tese de Doutorado

### Capítulo 1 - Introdução
- **Seção 1.1**: Quinases Proteicas - Arquitetura Molecular e Relevância Terapêutica
  - 518 quinases humanas (taxonomia Manning et al.)
  - Centralidade topológica em redes de sinalização celular
  - 72 inibidores aprovados, $80B anuais
  
- **Seção 1.2**: O Paradoxo da Seletividade - Desafio Farmacológico Fundamental
  - Homologia estrutural extraordinária (RMSD < 2Å)
  - Polifarmacologia e toxicidades dose-limitantes
  - Mutações de resistência (T790M em EGFR, T315I em BCR-ABL)

- **Seção 1.3**: Formulação do Problema Computacional
  - Três critérios formais: Precisão (R² > 0.7), Escalabilidade (>10⁶ pred/hora), Cobertura Universal
  - Definição matemática da função de afinidade φ: K × C → ℝ

- **Seção 1.4**: Hipótese Central - Primado da Sequência
  - Abandono de representações geométricas 3D
  - Reformulação semântica via PLMs
  - Vantagens: Universalidade, Contextualidade, Escalabilidade

- **Seção 1.5**: DT-Kinase - Objetivos e Contribuições
  - Arquitetura proposta: ESM-2 + SMI-TED + CNN + Cross-Attention
  - Predição multi-tarefa (classificação + regressão)

### Capítulo 2 - Estado da Arte
- Paradigmas experimentais de perfilagem (Karaman 2008, Klaeger 2017)
- Limitações econômicas (~$30-40K por composto para painel completo)
- Problema do "quinoma escuro" (~200 quinases sem ensaios validados)
- Métodos de docking: erro sistemático ±2-3 kcal/mol vs 1.4 kcal/mol necessário
- DeepDTA e limitações de fusão tardia
- Modelos de Linguagem de Proteínas como paradigma emergente

### Capítulo 3 - Fundamentos Teóricos
- Arquitetura Transformer e auto-atenção (eq. 3.2)
- ESM-2: Masked Language Modeling em 65M sequências
- SMI-TED: Representação molecular via SMILES
- Mecanismo de Atenção Cruzada (eq. 3.6)
- Framework Multi-Tarefa com loss combinada (eq. 3.10)
- Resolução teórica do paradoxo de seletividade

---

## ✅ Checklist Conceitual (Alinhado com Tese)

- [ ] **semantic-screening** = Plataforma aberta e modular (independência estrutural, escalabilidade computacional)
- [ ] **DT-Kinase** = Arquitetura neural específica (ESM-2 + SMI-TED + CNN + Cross-Attention)
- [ ] **Paradoxo de Seletividade** = 518 quinases com RMSD < 2Å nos sítios ATP requerem abordagem semântica
- [ ] **Critérios de Design** (Cap. 2): R² > 0.7, Throughput > 10⁶/hora, Cobertura 100% quinoma
- [ ] **Hipótese Central** (Cap. 1.4): Primado da sequência—informacão estrutural está codificada em sequência via PLMs
- [ ] **Resolução Teórica** (Cap. 3.8): Contextualidade global + história evolutiva + correspondências posição-específicas
- [ ] Ambos operam sem requerer estrutura 3D experimental
- [ ] semantic-screening fornece infra de embeddings, validação, estratificação
- [ ] DT-Kinase fornece arquitetura CNN + Cross-Attention para interações proteína-ligante

---

**Última atualização**: Janeiro 2026  
**Documento**: Esclarecimento conceitual  
**Status**: Referência para toda documentação
