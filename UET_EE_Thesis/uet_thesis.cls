%%
%% This file uet_thesis.cls is based on `Thesis.cls' and 'ECSthesis.cls', by    %% Sunil Patel and Steve R. Gunn respectively.
%%
%% Created by Steve R. Gunn, first modified by Sunil Patel: www.sunilpatel.co.uk %% and then adopted for UET style by M Tahir (mtahir@uet.edu.pk).

\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{uet_thesis}
              [2010/22/03 v1.1
   LaTeX document class]
\def\baseclass{book}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\def\@checkoptions#1#2{
  \edef\@curroptions{\@ptionlist{\@currname.\@currext}}
  \@tempswafalse
  \@tfor\@this:=#2\do{
    \@expandtwoargs\in@{,\@this,}{,\@curroptions,}
    \ifin@ \@tempswatrue \@break@tfor \fi}
  \let\@this\@empty
  \if@tempswa \else \PassOptionsToClass{#1}{\baseclass}\fi
}
\@checkoptions{11pt}{{10pt}{11pt}{12pt}}
\PassOptionsToClass{a4paper}{\baseclass}
\ProcessOptions\relax
\LoadClass{\baseclass}
\newcommand\bhrule{\typeout{-------------------------------------------}}

\newcommand\Declaration[1]{
\btypeout{Declaration of Authorship}
%\addtotoc{Declaration of Authorship}
\thispagestyle{plain}
\null\vfil
%\vskip 60\p@
\begin{center}{\huge \bf Declaration \par}\end{center}
%\vskip 60\p@
{\normalsize #1}
\vfil\vfil\null
%\cleardoublepage
}


\newcommand\btypeout[1]{\bhrule\typeout{\space #1}\bhrule}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space \number\year}
\usepackage{setspace}
\onehalfspacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{2.0ex plus0.5ex minus0.2ex}
\usepackage{vmargin}
\setmarginsrb           { 1.5in}  % left margin
                        { 0.6in}  % top margin
                        { 1.0in}  % right margin
                        { 0.8in}  % bottom margin
                        {  20pt}  % head height
                        {0.25in}  % head sep
                        {   9pt}  % foot height
                        { 0.3in}  % foot sep
\raggedbottom
\setlength{\topskip}{1\topskip \@plus 5\p@}
\doublehyphendemerits=10000       % No consecutive line hyphens.
\brokenpenalty=10000              % No broken words across columns/pages.
\widowpenalty=9999                % Almost no widows at bottom of page.
\clubpenalty=9999                 % Almost no orphans at top of page.
\interfootnotelinepenalty=9999    % Almost never break footnotes.
\usepackage{fancyhdr}
\lhead[\rm\thepage]{\fancyplain{}{\sl{\rightmark}}}
\rhead[\fancyplain{}{\sl{\leftmark}}]{\rm\thepage}
\chead{}\lfoot{}\rfoot{}\cfoot{}
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\btypeout{\thechapter\space #1}\markboth{\@chapapp\ \thechapter\ #1}{\@chapapp\ \thechapter\ #1}}
\renewcommand{\sectionmark}[1]{}
\renewcommand{\subsectionmark}[1]{}
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}
\usepackage{amsmath,amsfonts,amssymb,amscd,amsthm,xspace}
\theoremstyle{plain}
\newtheorem{example}{Example}[chapter]
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{axiom}[theorem]{Axiom}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\usepackage[centerlast,small,sc]{caption}
\setlength{\captionmargin}{20pt}
\newcommand{\fref}[1]{Figure~\ref{#1}}
\newcommand{\tref}[1]{Table~\ref{#1}}
\newcommand{\eref}[1]{Equation~\ref{#1}}
\newcommand{\cref}[1]{Chapter~\ref{#1}}
\newcommand{\sref}[1]{Section~\ref{#1}}
\newcommand{\aref}[1]{Appendix~\ref{#1}}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\dbltopfraction}{.85}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{.75}
\setcounter{topnumber}{9}
\setcounter{bottomnumber}{9}
\setcounter{totalnumber}{20}
\setcounter{dbltopnumber}{9}
\usepackage{graphicx}
%\usepackage{epstopdf}
\usepackage[scriptsize]{subfigure}
\usepackage{booktabs}
\usepackage{rotating}
\usepackage{listings}
%\usepackage{lstpatch}
\lstset{captionpos=b,
        frame=tb,
        basicstyle=\scriptsize\ttfamily,
        showstringspaces=false,
        keepspaces=true}
\lstdefinestyle{matlab} {
        language=Matlab,
        keywordstyle=\color{blue},
        commentstyle=\color[rgb]{0.13,0.55,0.13}\em,
        stringstyle=\color[rgb]{0.7,0,0} }
\usepackage[pdfpagemode={UseOutlines},bookmarks=true,bookmarksopen=true,
   bookmarksopenlevel=0,bookmarksnumbered=true,hypertexnames=false,
   colorlinks,linkcolor={blue},citecolor={blue},urlcolor={red},
   pdfstartview={FitV},unicode,breaklinks=true]{hyperref}
\pdfstringdefDisableCommands{
   \let\\\space
}
\newcommand*{\supervisor}[1]{\def\supname{#1}}
\newcommand*{\examiner}[1]{\def\examname{#1}}
\newcommand*{\degree}[1]{\def\degreename{#1}}
\newcommand*{\authors}[1]{\def\authornames{#1}}
\newcommand*{\advisor}[1]{\def\advisorname{#1}}
\newcommand*{\session}[1]{\def\groupsession{#1}}
\newcommand*{\addresses}[1]{\def\addressnames{#1}}
\newcommand*{\university}[1]{\def\univname{#1}}
\newcommand*{\UNIVERSITY}[1]{\def\UNIVNAME{#1}}
\newcommand*{\department}[1]{\def\deptname{#1}}
\newcommand*{\DEPARTMENT}[1]{\def\DEPTNAME{#1}}
\newcommand*{\group}[1]{\def\groupname{#1}}
\newcommand*{\GROUP}[1]{\def\GROUPNAME{#1}}
\newcommand*{\faculty}[1]{\def\facname{#1}}
\newcommand*{\FACULTY}[1]{\def\FACNAME{#1}}
\newcommand*{\subject}[1]{\def\subjectname{#1}}
\newcommand*{\keywords}[1]{\def\keywordnames{#1}}
\supervisor  {}
\examiner    {}
\degree      {}
\authors     {}
\university  {University of Engineering and Technology Lahore}
\UNIVERSITY  {UNIVERSITY OF ENGINEERING AND TECHNOLOGY LAHORE}
\department  {{Department of Electrical Engineering}}
\DEPARTMENT  {{DEPARTMENT OF ELECTRICAL ENGINEERING}}

\group       {{Research Group Name}}
\GROUP       {{RESEARCH GROUP NAME (IN BLOCK CAPITALS)}}
\faculty     {{Faculty Name}}
\FACULTY     {{FACULTY NAME (IN BLOCK CAPITALS)}}
\addresses   {}
\subject     {}
\keywords    {}
\renewcommand\maketitle{
  \btypeout{Title Page}
  \hypersetup{pdftitle={\@title}}
  \hypersetup{pdfsubject=\subjectname}
  \hypersetup{pdfauthor=\authornames}
  \hypersetup{pdfkeywords=\keywordnames}
  \thispagestyle{empty}
  \begin{titlepage}
    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
    \setcounter{footnote}{0}
    \null\vfil
    \vskip 60\p@
    \begin{center}
      \setlength{\parskip}{0pt}
      {\huge \bf \@title \par}
%		{\rule{35em}{1.5pt}}
      \vfill
		\begin{figure}[h]
  		\centering
    	\includegraphics{./Figures/uet_logo.pdf}
 		\end{figure}
		\vfill
%      {\Large {\bf Session:} \groupsession \par}
%      \bigskip
		{\Large {\bf Submitted by:} \\ {\authornames} \par}
		\bigskip
		\bigskip
      \bigskip

		{\Large {\bf Supervised by:} \advisorname \par}
      \vfill
%%      {\large A thesis submitted in partial fulfillment for the \par}
%%      {\large degree of Doctor of Philosophy \par}
      \bigskip
		\bigskip
%%      {\large in the \par}
%		{\rule{35em}{1.5pt}}
      {\Large \deptname \par}
      {\Large\textbf{\univname}\par}
%      \bigskip

%      {\Large \@date \par}
%      \bigskip
    \end{center}
%    \par
%    \@thanks
%    \vfil\null
  \end{titlepage}
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
%  \global\let\@title\@empty
%  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  \cleardoublepage
}
\newenvironment{abstract}
{
  \btypeout{Abstract Page}
  \thispagestyle{empty}
  \begin{center}
    \setlength{\parskip}{0pt}
    {\huge{ \bf Abstract} \par}
   \end{center}
}
{
  \vfil\vfil\vfil\null
  \cleardoublepage
}
\addtocounter{secnumdepth}{1}
\setcounter{tocdepth}{6}
\newcounter{dummy}
\newcommand\addtotoc[1]{
\refstepcounter{dummy}
\addcontentsline{toc}{chapter}{#1}}
\renewcommand\tableofcontents{
\btypeout{Table of Contents}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}
    \@starttoc{toc}
    \if@restonecol\twocolumn\fi
   \cleardoublepage
}\end{spacing}
}
\renewcommand\listoffigures{
\btypeout{List of Figures}
\addtotoc{List of Figures}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}
              {\MakeUppercase\listfigurename}}
    \@starttoc{lof}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\renewcommand\listoftables{
\btypeout{List of Tables}
\addtotoc{List of Tables}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}
    \@starttoc{lot}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\newcommand\listsymbolname{Abbreviations}
\usepackage{longtable}
\newcommand\listofsymbols[2]{
\btypeout{\listsymbolname}
\addtotoc{\listsymbolname}
    \chapter*{\listsymbolname
      \@mkboth{
          \MakeUppercase\listsymbolname}{\MakeUppercase\listsymbolname}}
\begin{longtable}[c]{#1}#2\end{longtable}\par
    \cleardoublepage
}
\newcommand\listconstants{Physical Constants}
\usepackage{longtable}
\newcommand\listofconstants[2]{
\btypeout{\listconstants}
\addtotoc{\listconstants}
    \chapter*{\listconstants
      \@mkboth{
          \MakeUppercase\listconstants}{\MakeUppercase\listconstants}}
\begin{longtable}[c]{#1}#2\end{longtable}\par
    \cleardoublepage
}
\newcommand\listnomenclature{Symbols}
\usepackage{longtable}
\newcommand\listofnomenclature[2]{
\btypeout{\listnomenclature}
\addtotoc{\listnomenclature}
    \chapter*{\listnomenclature
      \@mkboth{
          \MakeUppercase\listnomenclature}{\MakeUppercase\listnomenclature}}
\begin{longtable}[c]{#1}#2\end{longtable}\par
    \cleardoublepage
}
\newcommand\acknowledgements[1]{
\btypeout{Acknowledgments}
\addtotoc{Acknowledgments}
\thispagestyle{plain}
\vspace{0.5in}
\begin{center}{\huge{\bf Acknowledgments} \par}\end{center}
{\normalsize #1}
\vfil\vfil\null
}

\newcommand\CertificationBSc[1]{
\btypeout{Certification}
%\addtotoc{Declaration of Authorship}
\thispagestyle{plain}
\null\vfil
%\vskip 60\p@
\begin{center}{\huge \bf \@title \par}\end{center}
\vskip 60\p@
\begin{center}
Submitted to the faculty of Electrical Engineering \\ of the University of Engineering and Technology Lahore \\ in partial fulfillment of the requirements for the Degree of \\ \LARGE{Bachelor of Science} \\ \Large{in} \\ \LARGE{\bf Electrical Engineering}.
\end{center}
{\normalsize #1}
\vfil\vfil\null
\bigskip
\bigskip

\begin{table}[h]
\centering
\begin{tabular}{ccccc}
\rule[0em]{15em}{1.0pt} & ~~~~~~~~~~~~~~ & \rule[0em]{15em}{1.0pt} \\
Internal Examiner & ~~~~~~~~~~~~~~ & External Examiner
\end{tabular}
\end{table}

\vfill

\begin{table}[h]
\centering
\begin{tabular}{ccccc}
 & \rule[0em]{15em}{1.0pt} &  \\
 & Director &  \\
 & Undergraduate Studies &
\end{tabular}
\end{table}

\vfill

\begin{center}
\bigskip

{\Large \deptname \par}
{\Large \textbf{\univname}\par}

\end{center}
%\cleardoublepage
}

\newcommand\CertificationMSc[1]{
\btypeout{Certification}
%\addtotoc{Declaration of Authorship}
\thispagestyle{empty}
\null\vfil
%\vskip 60\p@
\begin{center}{\huge \bf \@title \par}\end{center}
\vskip 60\p@
\begin{center}
Submitted to the faculty of the Electrical Engineering Department \\ of the University of Engineering and Technology Lahore \\ in partial fulfillment of the requirements for the Degree of \\ \LARGE{Master of Science} \\ \Large{in} \\ \LARGE{\bf Electrical Engineering}.
\end{center}
{\normalsize #1}
\vfil\vfil\null
\bigskip
\bigskip

\begin{table}[h]
\centering
\begin{tabular}{ccccc}
\rule[0em]{15em}{1.0pt} & ~~~~~~~~~~~~~~ & \rule[0em]{15em}{1.0pt} \\
Internal Examiner & ~~~~~~~~~~~~~~ & External Examiner
\end{tabular}
\end{table}

\vfill

\begin{table}[h]
\centering
\begin{tabular}{ccccc}
\rule[0em]{15em}{1.0pt} & ~~~~~~~~~~~~~~ & \rule[0em]{15em}{1.0pt} \\
Dean & ~~~~~~~~~~~~~~ & Chairman \\
Faculty of Electrical Engineering & ~~~~~~~~~~~~~~ & Electrical Engineering Department
\end{tabular}
\end{table}

\vfill

\begin{center}
	
\bigskip

{\Large \deptname \par}
{\Large \textbf{\univname}\par}

\end{center}
%\cleardoublepage
}

\newcommand\dedicatory[1]{
\btypeout{Dedicatory}
\thispagestyle{plain}
\null\vfil
\vskip 60\p@
\begin{center}{\Large \sl #1}\end{center}
\vfil\null
\cleardoublepage
}

\renewcommand\backmatter{
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \renewcommand{\bibname}{References}
  \addtotoc{\bibname}
  \btypeout{\bibname}
  \@mainmatterfalse}
\endinput
%%
%% End of file `uet_thesis.cls'.
