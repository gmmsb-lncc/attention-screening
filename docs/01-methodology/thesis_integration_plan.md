# Plano de integração na tese: textos honestos e coerentes

Documento de execução para integrar o toolkit estatístico
(`scripts/statistical_analysis/` + `statistical_protocol.md`) na tese
LaTeX em `~/PhD/tex/`. Princípio orientador: **aditivo, nunca
sobrescrever**. O texto existente da tese já contém TOST com δ=0,05,
contagem 4 estritos + 4 fronteiriços + 1 não-equivalente, pré-registro
retroativo, σ amostral, bootstrap B=10⁴ — tudo permanece. As novas
camadas (limites empíricos, Hedges' g, post-hoc, TOST sensitivity,
RM-ANOVA + Tukey HSD com assumption checks, simultaneous CI plot,
MCSim heatmap) entram como **enriquecimento** documentado em apêndice
dedicado, com referências cirúrgicas no corpo principal.

## Princípios de honestidade (não negociáveis)

1. **Auditoria retroativa, não prospectiva.** O texto deve dizer
   explicitamente que o padrão Ash/Wognum 2025 foi adotado **após**
   coleta dos dados, e que o toolkit foi implementado pós-execução.
   Nada sugerindo que a tese foi pré-registrada sob o padrão.
2. **D1 fechada por escopo, não por cumprimento.** Sempre que D1
   aparecer, deixar claro que 5×5 CV está agendado para a versão de
   periódico (pós-defesa), e que a tese é robusta sob a cláusula de
   transparência (Item 5 da Conclusão do paper).
3. **DT-Kinase pior em AUROC/AUPRC.** O Tukey HSD rejeita pares onde
   DT-Kinase é estatisticamente inferior a DrugBAN/ConPLex em
   AUROC/AUPRC. Reportar honestamente; não esconder atrás de "MCC
   primária".
4. **r@p=0,8 instabilidade.** Caracterizar como "característica
   operacional do modelo sob restrição de precisão alta", não como
   "vantagem" nem como "falha". É o que o dado mostra.
5. **Convergência tripla é o ponto forte.** A H1 ganha solidez quando
   bootstrap + ANOVA + TOST concordam — frase a frase, citar as três
   camadas em vez de só o bootstrap.
6. **Cláusula de transparência.** Citar Item 5 do paper textualmente
   onde declarar D1: o próprio Ash/Wognum endossa desvios documentados.

## Estado atual da tese (mapeado em 2026-05-08)

| Já existe | Localização | Mantido inalterado? |
|---|---|---|
| H1 + H2 + TOST critério + SESOI + pré-registro retroativo | `introducao.tex` ~linhas 50–58 | sim |
| Bootstrap B=10⁴ + Holm-Bonferroni + TOST + pré-registro retroativo | `capitulo4.tex` `\paragraph{Natureza estatística dos intervalos reportados}` (linha 643) | sim, com adendo |
| Tabela `tab:pareados-all` + contagem H1 (4 estrito + 4 fronteiriço + 1 não-eq.) | `capitulo5.tex` linha 486, 593, 710 | sim |
| Auditoria calibração Brier + ECE + reliability diagram | `anexoB.tex` linha 813+ | sim |
| Comitê 3 vs 4 modelos | `anexoB.tex` linha 916 | sim |
| 24 lições metodológicas | `apendiceF.tex` | sim, com Lição 25 nova |

| Falta | Onde adicionar |
|---|---|
| Padrão Ash/Wognum 2025 + 4 guidelines + 4/5 compliant + D1/D2/D3 | `capitulo4.tex` nova subseção; `anexoB.tex` nova seção principal |
| Limite inferior (null model) + limite superior (assay noise) | `anexoB.tex` nova seção; `capitulo5.tex` 1 parágrafo + 1 tabela contextualizando |
| Hedges' g paired J(4) | `anexoB.tex` nova seção (tabela por par × métrica × corpus) |
| Post-hoc classification (p@r, r@p, TNR@r) | `anexoB.tex` nova seção (tabela por modelo) |
| TOST sensitivity sobre 6 bandas | `anexoB.tex` nova seção (tabela 6 bandas × 3 corpora) |
| RM-ANOVA + Tukey HSD + assumption checks | `anexoB.tex` nova seção (tabela ANOVA + Tukey por métrica) |
| Simultaneous CI plot (12 figs) e MCSim heatmap (12 figs) | `anexoB.tex` figuras + cross-ref |
| Reporting checklist por corpus | `anexoB.tex` tabela compliance |
| Lição 25 (migração para Ash/Wognum) | `apendiceF.tex` ao final |
| Auditoria contra os 5 itens da Conclusão do paper | `conclusao.tex` 1 parágrafo |
| Frase de reconhecimento do padrão | `introducao.tex` final do parágrafo de pré-registro |

## Fases de execução (sequência rígida — cross-references resolvem)

### Fase 1 — Apêndice B: nova seção "Auditoria estatística complementar"

Local: `~/PhD/tex/anexoB.tex`, ao final do capítulo (depois de §"Comitê
3-modelos vs 4-modelos" linha 916+).

Estrutura:

```latex
\section{Auditoria estatística complementar (Ash/Wognum 2025)}\label{sec:auditoria-stats-aw}

\subsection{Padrão adotado e auditoria retroativa}\label{sec:auditoria-stats-padrao}
% 1 parágrafo introduzindo o paper, 4 guidelines, 5 itens da conclusão.
% Frase-chave: "adoção pós-coleta dos dados, conforme cláusula de
% transparência (Item 5 da Conclusão)."

\subsection{Limites empíricos por corpus}\label{sec:auditoria-stats-limites}
% Lower: null-model majority-class. Upper: Brown 2009 / Kramer 2012
% sob ruído IC50 log10(2)=0,301. Tabela: corpus x {null, upper, range
% observado, gap até teto}.
% Mensagem: "modelos ocupam <15% do espaço de melhoria; gap até teto >
% gap entre arquiteturas".

\subsection{Effect size pareado (Hedges' g com correção J(4))}\label{sec:auditoria-stats-hedges}
% Tabela 4 modelos × 4 modelos × 4 métricas × 3 corpora (resumida em
% mostrar apenas DT-Kinase vs cada baseline). Hedges' g paired J(4).
% Footnote: J(4) = 1 - 3/15 = 0.8 sob aproximação Lakens 2013 / Borenstein.

\subsection{Análise de sensibilidade TOST}\label{sec:auditoria-stats-tost-sensitivity}
% Tabela 6 bandas × 3 corpora × {n_equivalente / 6}. Banda 0,05
% marcada PRIMARY. Bandas Cohen-anchored: 0,2σ, 0,5σ, 0,8σ.
% Mensagem: "robustez do veredicto sob escolhas alternativas de banda".

\subsection{RM-ANOVA, Tukey HSD e checagens de premissa}\label{sec:auditoria-stats-anova}
% Tabela ANOVA p + p_bonf + Tukey HSD pairwise por métrica × corpus.
% Bloco de assumption checks: Shapiro-Wilk per modelo, Levene
% across-modelos, Levene em diferenças par-a-par (proxy de
% esfericidade).
% Mensagem: "convergência com bootstrap em MCC (NS) e divergência em
% AUROC/AUPRC (sig)."

\subsection{Métricas pós-classificação}\label{sec:auditoria-stats-posthoc}
% Tabela: 4 modelos × {precision@recall=0.8, recall@precision=0.8,
% TNR@recall=0.9} × 3 corpora.
% Destaque: r@p=0.8 do DT-Kinase com std alta -> "característica
% operacional sob restrição de precision alta".

\subsection{Visualizações pareadas}\label{sec:auditoria-stats-figs}
% Figs Simultaneous CI + MCSim heatmap por corpus × métrica.
% Cross-ref com captions explicativas.

\subsection{Reporting checklist por corpus}\label{sec:auditoria-stats-checklist}
% Tabela 10 itens × 3 corpora com PASS/FAIL automatizado.
% Cita statistical_protocol.md §4.

\subsection{Implementação}\label{sec:auditoria-stats-impl}
% Lista os scripts em scripts/statistical_analysis/, entry-points
% (run_full_stats.sh, run_all_corpora.sh), suite de testes.
```

**Tabelas a incluir (extraídas dos JSONs em `results/statistical/`):**

| Tabela | Origem | Caption guideline |
|---|---|---|
| `tab:limites-empiricos` | `null_model.json` + `upper_limit.json` | "Limite inferior: classificador majority-class. Limite superior: ruído experimental IC₅₀ 2-fold (Brown 2009)." |
| `tab:hedges-g-paired` | `effect_size.json` | "g pareado com correção Hedges J(4)=0,8 (Lakens 2013); cutoffs 0,2/0,5/0,8 (Cohen 1988)." |
| `tab:tost-sensitivity` | `tost.json` | "Contagem de pares equivalentes (de 6 unord.) por banda. δ=0,05 marcado como primário SESOI." |
| `tab:anova-tukey` | `anova_tukey.json` | "ANOVA RM por métrica + Bonferroni inter-métrica (m=4) + Tukey HSD pairwise + checagens de premissa (Shapiro-Wilk, Levene)." |
| `tab:posthoc-classification` | `posthoc.json` | "Métricas operacionais por modelo. precision@recall=0,8 e recall@precision=0,8 capturam regimes opostos de operação." |
| `tab:reporting-checklist` | `checklist.md` | "Conformidade com a reporting checklist do `statistical_protocol.md` §4." |
| `panel.tex` | `panel.tex` | já gerado por `aggregate_panel.py` (caption explícita já inclui convenção σ + ddof=1 + Brown/Kramer cite). |

**Figuras a incluir** (12 + 12 = 24 PDFs em `results/statistical/{corpus}/figures/`):

- `\includegraphics{results/statistical/non_human/figures/sim_ci_mcc.pdf}` — etc.
- Estratégia: 1 figura combinada por métrica (3 corpora lado a lado em sub-painéis), totalizando 4 figuras simultâneo + 4 MCSim. Reduz 24 → 8.

### Fase 2 — Cap. 4: nova subseção "Padrão estatístico Ash/Wognum 2025"

Local: `~/PhD/tex/capitulo4.tex` ao final da seção §"Refinamentos
estatísticos" ou imediatamente antes de Cap. 5 (após linha 643+).

```latex
\subsection{Padrão estatístico Ash/Wognum 2025 e desvios declarados}\label{sec:ashwognum}

% 1 parágrafo: introduzir o paper, 4 guidelines, decisão de adoção
% RETROATIVA pós-coleta dos dados.
% Frase-chave: "auditoria empreendida em 2026-05-07; o protocolo aqui
% reportado mantém-se na forma em que os dados foram efetivamente
% coletados, com adoção retroativa do padrão e declaração explícita
% de desvios sob a cláusula de transparência (Item 5 da Conclusão)."

\subsubsection{Desvios declarados (D1, D2, D3)}\label{sec:ashwognum-desvios}

% Subitem D1: single split + 5 sementes vs 5x5 CV.
% Justificativa: custo de retreino. Marcação: "thesis-scope only".
% Migração para 5x5 CV agendada para versão de periódico, pós-defesa.

% Subitem D2: bootstrap pareado por proteína primário.
% RM-ANOVA + Tukey HSD camada complementar (n=5 subdimensiona ANOVA).
% Convergência reportada lado a lado.

% Subitem D3: banda TOST δ=0,05 MCC SESOI-anchored primária.
% Bandas Cohen-d-anchored (Lakens 2017) reportadas em sensibilidade
% (Apêndice B§\ref{sec:auditoria-stats-tost-sensitivity}).

\subsubsection{Camadas complementares adicionadas}\label{sec:ashwognum-camadas}

% 1 parágrafo enumerando: null-model lower limit, upper limit
% (Brown/Kramer), Hedges' g paired J(4), post-hoc classification
% (precision@recall, recall@precision, TNR@recall), TOST sensitivity
% sobre 6 bandas, RM-ANOVA + Tukey HSD com Shapiro/Levene,
% simultaneous CI plot, MCSim heatmap.
% Cross-ref para Apêndice B§\ref{sec:auditoria-stats-aw}.
```

### Fase 3 — Cap. 5: enriquecimentos cirúrgicos

Local: `~/PhD/tex/capitulo5.tex`. **Não reescrever nada existente.**
Adicionar 3 parágrafos pontuais:

3.1. **No início de §`sec:resultados-baselines` (linha 106 e seguintes),
após a introdução** — 1 parágrafo:

```latex
\paragraph{Camadas estatísticas complementares.} % NOVO
% Antes de prosseguir com as métricas pareadas, registra-se que esta
% seção é lida sob cinco camadas complementares: (i) bootstrap pareado
% por proteína (B=10^4) reportado nas Tabelas...; (ii) RM-ANOVA por
% métrica com correção Bonferroni (m=4) seguida de Tukey HSD pairwise,
% Apêndice~\ref{sec:auditoria-stats-anova}; (iii) effect size pareado
% via Hedges' g com correção J(4)=0,8 para n=5,
% Apêndice~\ref{sec:auditoria-stats-hedges}; (iv) análise de
% sensibilidade do TOST sobre seis bandas δ_eq,
% Apêndice~\ref{sec:auditoria-stats-tost-sensitivity}; (v) métricas
% pós-classificação operacionalmente relevantes,
% Apêndice~\ref{sec:auditoria-stats-posthoc}. As três camadas
% inferenciais (bootstrap, ANOVA+Tukey, TOST) convergem na maioria dos
% pares, fortalecendo a leitura H1.
```

3.2. **Após §"Equilíbrio entre quatro arquiteturas distintas" (linha
593+)** — 1 parágrafo de convergência tripla:

```latex
\paragraph{Convergência metodológica entre camadas inferenciais.} % NOVO
% Sob MCC primária, as três camadas concordam: bootstrap pareado tem
% IC95 cruzando zero ou na fronteira, RM-ANOVA com Bonferroni inter-
% métrica retorna p_bonf > 0,05 (não rejeita), e o TOST com banda
% δ=0,05 confirma equivalência estrita ou fronteiriça nos pares
% envolvendo DT-Kinase. Sob AUROC e AUPRC, todas as três camadas
% concordam que DT-Kinase difere significativamente de DrugBAN e
% ConPLex (Tukey HSD rejeita; Hedges' g paired ∈ [-1,32; -2,62],
% magnitude grande pelo cutoff de Cohen). A leitura honesta é que H1
% se sustenta em MCC mas não em AUROC/AUPRC; a tese declara MCC como
% métrica primária e a equivalência é bem-formada nessa escolha
% (Capítulo~\ref{cap-1-introducao}, Seção~\ref{sec:rqs-escopo}).
```

3.3. **Após §"Validação no Corpus Geral" ou no início de
§"Síntese da Comparação"** — 1 parágrafo de limites empíricos:

```latex
\paragraph{Patamar comum dos quatro modelos contextualizado.} % NOVO
% O classificador majority-class produz MCC = 0 por construção (limite
% inferior do problema). O ruído experimental IC50 de 2-fold (Brown
% et al. 2009; Kramer et al. 2012, σ_log10 ≈ 0,301) implica MCC
% ceiling de aproximadamente 0,876 no Não-Humano (limite superior).
% Os quatro modelos avaliados ocupam a faixa [0,491; 0,544] em MCC,
% i.e., aproximadamente 60 % do gap até o limite superior permanece
% inexplorado. A diferença entre arquiteturas (≈0,05 MCC) é uma fração
% pequena do gap até o teto teórico (≈0,38 MCC); a oportunidade
% científica relevante é fechar a distância para 0,876, não otimizar
% entre as quatro alternativas atuais
% (Apêndice~\ref{sec:auditoria-stats-limites}).
```

### Fase 4 — Apêndice F: Lição 25

Local: `~/PhD/tex/apendiceF.tex` ao final.

```latex
\section{Lição 25: Adoção do padrão Ash/Wognum 2025}\label{sec:licao-25}

% ~70 linhas baseadas no §6.15 do licoes_aprendidas.md já escrito.
% Conteúdo: confronto contra os 5 itens da Conclusão; três
% divergências D1/D2/D3 declaradas; três camadas de migração
% corretiva (declaração explícita, camadas complementares, mudança de
% visualização); enunciado da lição.
```

Texto-base já está em `docs/01-methodology/licoes_aprendidas.md` §6.15
— traduzir LaTeX direto.

### Fase 5 — Conclusão: parágrafo de auditoria

Local: `~/PhD/tex/conclusao.tex` ao final, antes do parágrafo final de
trabalhos futuros.

```latex
\paragraph{Auditoria contra Ash, Wognum et al. 2025.} % NOVO
% Concluído em 2026-05-08, o protocolo estatístico desta tese foi
% retrospectivamente confrontado contra a Perspective de Ash, Wognum,
% Rodríguez-Pérez et al. (J. Chem. Inf. Model. 65:9398-9411, 2025).
% Dos cinco itens enumerados na Conclusão do paper, quatro são
% plenamente cumpridos (Tukey HSD para pares com checagem de premissa,
% significância prática via Cohen's d e limites de performance,
% visualização de comparações pareadas, transparência sobre desvios);
% um é desvio explicitamente declarado (D1: 5 sementes em split fixo
% em vez de 5x5 CV recomendado, mantido por escopo de tese e
% migrável para a versão de periódico). A migração executada está
% documentada no Apêndice~\ref{sec:auditoria-stats-aw} e no
% Apêndice~\ref{sec:licao-25}; o toolkit de análise é open-source no
% repositório atestação-screening.
```

### Fase 6 — Introdução: frase de cross-ref (opcional)

Local: `~/PhD/tex/introducao.tex` ao final do parágrafo de
"Pré-registro retroativo" (linha 58).

```latex
% Adicionar como última frase do parágrafo:
Adicionalmente, o protocolo estatístico foi auditado retrospectivamente
contra a recomendação de Ash, Wognum et al. (2025) e migrado para
conformidade nos itens praticáveis sob as restrições experimentais
deste trabalho; quatro de cinco itens enumerados na Conclusão do paper
são plenamente cumpridos, com um desvio (D1: 5 sementes em split fixo)
declarado em
$\S$\ref{sec:ashwognum-desvios}~do Capítulo~\ref{cap-4-metodologia} e
auditado em $\S$\ref{sec:auditoria-stats-aw}~do Apêndice~\ref{anexoB}.
```

## Verificação end-to-end

Após cada fase:

1. `cd ~/PhD && pdflatex -interaction=nonstopmode tese_lncc.tex` →
   sem erros LaTeX.
2. `bibtex tese_lncc.aux && pdflatex -interaction=nonstopmode tese_lncc.tex`
   (×2 para resolver crossrefs).
3. Grep por `??` no PDF gerado: `pdftotext tese_lncc.pdf - | grep -c '??'`
   — deve ser 0 (zero refs quebradas).
4. Spot-check de 5 valores numéricos:
   - DT-Kinase NH MCC: panel.tex deve casar com Cap. 5 narrativa
     (0,5007 ± 0,0118 σ; ou 0,506 ± 0,012 sob legacy convertido).
   - Null limit MCC = 0,000 (todos corpora).
   - Upper limit MCC NH = 0,876.
   - TOST primary NH 3/6 (smoke B=2000) ou valor canônico após B=10⁴.
   - ANOVA p_bonf MCC NH ≈ 0,21 (NS).
5. Auditoria de honestidade — ler cada novo parágrafo e marcar:
   - Há claim que excede o que o dado mostra?
   - Há reference a "vantagem" ou "superioridade" sem suporte?
   - D1 está declarada como thesis-scope?
   - Pré-registro retroativo está mantido?

## Numeração final dos artefatos vinculáveis

Para garantir que cada novo objeto LaTeX tem `\label` único:

```
\label{sec:ashwognum}                    Cap.4 nova subseção
\label{sec:ashwognum-desvios}            Cap.4 sub-sub D1/D2/D3
\label{sec:ashwognum-camadas}            Cap.4 sub-sub camadas
\label{sec:auditoria-stats-aw}           Anexo B nova seção
\label{sec:auditoria-stats-padrao}       Anexo B padrão
\label{sec:auditoria-stats-limites}      Anexo B limites
\label{sec:auditoria-stats-hedges}       Anexo B Hedges
\label{sec:auditoria-stats-tost-sensitivity}  Anexo B TOST
\label{sec:auditoria-stats-anova}        Anexo B ANOVA
\label{sec:auditoria-stats-posthoc}      Anexo B post-hoc
\label{sec:auditoria-stats-figs}         Anexo B figs
\label{sec:auditoria-stats-checklist}    Anexo B checklist
\label{sec:auditoria-stats-impl}         Anexo B impl
\label{sec:licao-25}                     Apêndice F
\label{tab:limites-empiricos}            Tabela limites
\label{tab:hedges-g-paired}              Tabela Hedges
\label{tab:tost-sensitivity}             Tabela TOST sensitivity
\label{tab:anova-tukey}                  Tabela ANOVA + Tukey
\label{tab:posthoc-classification}       Tabela post-hoc
\label{tab:reporting-checklist}          Tabela checklist compliance
\label{tab:stat-panel-{human,non_human,all}}   Tabelas panel.tex (já existem)
\label{fig:sim-ci-{mcc,auroc,f1,auprc}}  4 figuras sim CI
\label{fig:mcsim-{mcc,auroc,f1,auprc}}   4 figuras MCSim
```

## Ordem de execução recomendada

1. **Aguardar B=10⁴ canônico do non_human** (em execução agora).
2. **Rodar B=10⁴ em human + all** (24h paralelas em d01/d02/d03; ou
   se preferir, rodar B=2000 overnight como smoke completo dos 3
   corpora primeiro).
3. **Fase 1** (Apêndice B nova seção) — fonte primária dos números.
4. **Fase 2** (Cap. 4 nova subseção) — declara o padrão e os desvios.
5. **Fase 3** (Cap. 5 enriquecimentos cirúrgicos) — 3 parágrafos.
6. **Fase 4** (Apêndice F Lição 25) — registro narrativo da migração.
7. **Fase 5** (Conclusão) — 1 parágrafo de auditoria.
8. **Fase 6** (Introdução) — opcional, 1 frase cross-ref.
9. **Verificação** end-to-end LaTeX + spot-check + auditoria de
   honestidade.
10. **Commit single** ao repositório de tese: `git -C ~/PhD add tex/ &&
    git -C ~/PhD commit -m "feat(stats): integrar auditoria
    Ash/Wognum 2025 (Apêndice B + Cap.4 §X.Y + Cap.5 + Conclusão +
    Apêndice F Lição 25)"`.

## Artefatos não-tese a sincronizar

- `~/PhD/figures/`: copiar 8 PDFs combinados (4 sim_ci + 4 mcsim) com
  `MIRROR_TO=~/PhD/figures bash scripts/statistical_analysis/run_full_stats.sh`
  (flag já implementada em `aggregate_panel.py`).
- Slides de defesa (`~/PhD/apresentacao_orientadores/`): após Fase 3,
  atualizar slide 23 ou criar slide novo dedicado a "Camadas
  estatísticas complementares" com a tabela dos 4 itens compliant +
  D1 declarado. Custo: ~30 min em Beamer.

## Custo total estimado

- Tabelas LaTeX manuais (extração JSON → tabela): ~3 h.
- Texto novo (parágrafos + Lição 25 + auditoria): ~4 h.
- Figuras integradas: ~1 h.
- Verificação compile + spot-check + auditoria honestidade: ~2 h.
- **Total: ~10 h trabalho de redação concentrado.**

## Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Cross-refs entre Cap. 4 e Apêndice B circulares | Fase 1 (apêndice) primeiro; Cap. 4 referencia o que já existe |
| Números do panel.tex não casam com narrativa Cap. 5 (legacy 0,506 ± 0,006 vs σ 0,012) | Convenção σ amostral declarada em §0; nota de rodapé na primeira ocorrência converte |
| Tabela 4×4 de Hedges' g muito grande | Reduzir a "DT-Kinase vs cada baseline" (12 células em vez de 48) |
| Compile LaTeX falha por figura ausente | Verificar que `MIRROR_TO=~/PhD/figures` foi executado antes de compile |
| Item 1 falha no AUROC/AUPRC contradiz "patamar comum" | Texto deve dizer "patamar comum em MCC; diferenciação em AUROC/AUPRC"; honesto |

## Decisão pendente do usuário antes de começar

1. **B=10⁴ em human + all primeiro**, ou **fechar non_human canônico
   e começar redação com B=2000 nos outros corpora**? (Diferença
   numérica mínima; impacto na confiança da tabela de sensibilidade.)
2. **Apêndice B existente é gigante** — a nova seção "Auditoria
   estatística complementar" deve virar **Apêndice G dedicado**?
   Alternativa: 50/50 com o atual — manter em B se quer integração com
   calibração; criar G se prefere camada separada.
3. **Tabelas LaTeX**: gerar manualmente ou estender
   `aggregate_panel.py` para emitir todas as 6 tabelas em `.tex`
   automaticamente? (Custo extra: ~2 h; ganho: regeneração trivial
   após B=10⁴ canônico final.)
