# a3 working notes (NC + special cavities) -- raw extracts from WebSearch (WebFetch/curl blocked)
Access: WebFetch -> getaddrinfo ENOTFOUND for every host (jacow, arxiv, wikipedia); curl -> proxy 403 for
jacow.org, accelconf.web.cern.ch, proceedings.jacow.org, epaper.kek.jp, arxiv.org, cds.cern.ch, osti.gov, journals.aps.org.
=> all facts below are from WebSearch result extracts (quoted passages), not full-text reading.

## PEP-II
- q1 links: EPAC1992_1289, PAC1993_1039, EPAC1994_2101, EPAC96 WEP069L/WEP070L, PRSTAB Oct 2000 (doaj cd9fde...)
- extract: 476 MHz, up to 150 kW wall; HER 0.93 MV -> ~122 kW; Ks = V^2/2P; Q ~ 31,000 with ports at 40C (EPAC92 table);
  3 rectangular WG + broadband loads; HOMs damped > 3 orders; nose pieces brazed, tuning by trimming nose length.
## EU HOM-damped 500 MHz (BESSY)
- links: e08 THXM03 talk/paper, PAC03 TPAB005, EPAC08 MOPP108, ESLS-RF 2004/2006/2007/2008 talks
- extract: undamped single cell optimised: Rsh 6.1 MOhm, R/Q 148.5 Ohm, Q0 41100; ESRF Al variant Q0~48000, Rs/Q0 ~150, Rs ~7.2 MOhm (Cu)
  3 circular double-ridge WG dampers, NiZn ferrite C48; cutoff 615 MHz (Al model); ESRF damper cutoff 452 MHz(?), <100 W in ferrite
  HOM Z < 5 kOhm long., 200 kOhm/m transverse; goals Rsh > 3-4 MOhm, < 0.7 m insertion, 100 kW
## MAX IV 100 MHz
- IPAC2011 MOPC051: NC all-copper capacity-loaded; theoretical Rsh 3.5 MOhm (practical 3.2); inner radius 410 mm, rod radius 60 mm,
  gap increased to 50 mm for 300 kV; 3rd harmonic Landau 5 MOhm
- ESLS 2015 MAX-lab: measured Rsh 3.3-3.45 MOhm (linac def), theory 3.52; Q0 19700-20500 vs theory 20923
- HarmonLIP 2025 slides: Rs0 1.71 MOhm, Q0 20284 (likely circuit def = half of linac: 3.42/2 = 1.71 -- INFERENCE)
## ESRF-EBS 352 MHz
- IPAC2021 MOPAB333 (D'Elia, Jacob, Serriere) doi 10.18429/JACoW-IPAC2021-MOPAB333; IPAC2011 MOPC004 -- derived from BESSY/ALBA
## ALS 500 MHz
- arXiv 1910.07175 (ALS digital LLRF): Q0 28670, QL 167, 499.64 MHz, 2 NC cavities ~1.3 MV total; R/Q listed "4.9 MOhm" (unit typo?)
- PAC1989 p121: f very sensitive to nose cone dims (URMEL/MAFIA); Rs transit-time corrected
- ALS-U 3HC: re-entrant, R ~2 MOhm, R/Q ~80 Ohm (IPAC2021 TUPAB053?), IPAC2019 WEPRB081 R/Q ~30 Ohm target (harmonic)
## KEK ARES
- 3 cavity pi/2 system: accel cavity + coupling cavity + TE013 storage cavity; accel cavity 4 rect WG (up/down) + grooved beam pipe both ends; SiC
- 508.887 MHz; damping-ring variant: Rsh/Q0 150 Ohm, Q0 ~30000 (sast2003 2CA-2?), SuperKEKB HOM power ~95 kW/cavity LER 9.4 A
## Elettra
- PAC1989 p220: 1 MV/cavity, 3.3 MV/m; "wide gap cavity" natural at 500 MHz ultra-rel; no dedicated HOM dampers; reused for Elettra 2.0
## Jensen CAS
- CAS Darmstadt09 slides: KEK PF 500 MHz example: optimised R/Q ~111 vs 107.5 Ohm, Q ~44270 vs 41630; nose cones raise T; round outer shape minimises losses
- arXiv 1601.05230 (RF cavity design): TM010 fundamental if l < 2.03076 R; rounding reduces field enhancement
## nose cone opt.
- LIBO arXiv 0712.0830: start from pillbox, nose cones reduce gap, raise T; cone angle ~20 deg optimal (Kilpatrick + shunt imp.); 8 mm hole compromise (3 GHz)
- ACS arXiv 1306.6287: 30 deg cone angle compromise (shunt impedance vs heat removal)
- LINAC2018 THPO042 (XiPAF nose cone buncher): MOO over cavity radius, gap length, cone angle at fixed f
## RF guns
- SLAC TN 04-4 (slac-tn-05-019) prototype BNL/SLAC/UCLA gun: full cell L 3.25 cm, D 8.35 cm; half cell L 2.28 cm, D 8.30 cm;
  iris D 2.50 cm, 0.95 cm radius of curvature on aperture; pi 2856.0, 0-mode 2852.5 MHz (3.5 MHz)
- LCLS-TN-05-3: racetrack arc radius 41.43 mm, arc separation 2.8 mm, racetrack cell length 34.01 mm, half cell radius 42.15 mm,
  half-cell length 24.26 mm, coupling slot width 13.2 mm; mode sep 15 MHz; 0-mode at cathode <3%
- PAC05 WPAT043: shunt impedance 49 MOhm/m, Q0 12000, QL 4000, 120 MV/m; PAC07 TUPMS047: Q0 13960 design / 13900 meas, 15.17 MHz
- arXiv 1503.05877: cathode field 120 (Gun III) / 140 MV/m (LCLS); full cell two offset circles 5.9 mm (racetrack)
- PITZ: 1.3 GHz, pi mode, coaxial coupler + doorknob; ~60 MV/m cathode, ~6.5 MW; up to 650-1000 us 10 Hz (arXiv 1803.05540)
## SLAC S-band
- two-mile Chapter_6: 2856 MHz, 2pi/3, t = 0.230 in (0.584 cm); shunt imp 53 MOhm/m (table); SLAC/AP-86: cavity 79 2a = 1.99 cm, 2b = 8.192 cm
- SuperB arXiv 1009.6178: period 3.499 cm, 86 cells, 120 deg
## CLIC-G (Grudiev & Wuensch LINAC10 MOP068; via researchgate extract)
- <a>/lambda 0.11; input/output iris radius 3.15/2.35 mm; iris thickness 1.67/1.00 mm; Q(Cu) first/last 5536/5738;
  R' first/last 81/103 MOhm/m; 26 regular cells; Es max 230 MV/m; 100 MV/m; eff 27.7%
## PEP-II (pinned)
- EPAC1992_1289: HER Ks 3.5 MOhm (V^2/2P), 0.93 MV, 122 kW wall, Q0 ~31000 with ports 40C; check 0.93^2/(2*122e3)=3.54 MOhm OK
  -> R/Q(circuit) ~113 Ohm, linac ~226 Ohm (INFERENCE)
- p99 MOP103 / HOM10 Rimmer: ~850 kV, 103 kW typical operation; rect HOM WG + broadband loads (PEP-II & DAFNE)
- PRSTAB 2000 (doaj cd9fde...): calc vs bench vs beam HOM impedances agree for strongest HOMs
## DAFNE 368.26 MHz
- e96 WEP032L / PAC93 0778: NC single cell, rounded, NO nose cones, long tapered tubes; 5 WGs (3 on cell, 2 on tapers); ~250 kV/ring
## DORIS: HOM Qext 300-10000 (Cornell SRF071120 slides)
## Choke mode
- KEK preprint 92-66 (Shintake): choke = notch filter trapping acc. mode, HOMs propagate radially to SiC; S-band HP test 50 MV/m (LINAC94 MO-91)
- IPAC2012 TUPPR015 (Zha, Shi, Grudiev, Wuensch, Tang...): CLIC 12 GHz choke: iris radii 2.35/3.15 mm, radial-line and choke gap 1 mm;
  radial choke because 2pi/3 cell too short for coaxial choke; IPAC2017: 11.424 GHz single-cell choke 75 MV/m limited by choke breakdown
- PRAB 22 031001 (2019) C-band choke-mode structures (SACLA?) -- small choke gap -> breakdown
## Crab / deflecting
- DQW (PRAB 21 082002 2018; arXiv 1805.08123): 400.79 MHz, R_T/Q 429.3 Ohm, G 87 Ohm; at 3.4 MV Bp ~73 mT, Ep ~37.6 MV/m; 1st HOM ~567-570 MHz;
  beam pipe separation 194 mm; SPS DQW: length 352 mm, width 288 mm, height w/o ports 286 mm (one source); Verdu-Andres: flange-flange 659.60,
  port-to-port height 495.71, width 329.60 mm. BNL PoP (arXiv 1502.04088): two QWRs sharing a load capacitor; R/Q 406, G 85, Ep 49 at 3.34 MV(?)
- RFD (SRF2023 TUPTB046 / arXiv 2307.06943): two ridged deflecting poles in square tank, TE11-like; tank transverse 281 mm (LHC) [rounded square];
  aperture 84 mm (FPC/HOM paper); Vt 3.34 MV, Ep 35.0 MV/m, Bp 55.1 mT, R_T/Q 431 Ohm(?); length & outer rounding radius tune f;
  pole length and END GAP optimised for shunt impedance; rounding at pole corners; curved gap -> sextupole compensation;
  operating 400.79 MHz, lowest dipole HOM 636 MHz, lowest acc HOM 699 MHz
- RFD PoP 400 MHz (PRSTAB 16 082001 2013): diameter 339.9 mm, length 542.4 mm
- RFD 1.3 GHz ILC (De Silva 2023 slides): pole separation 25 mm, aperture 50 mm, pole length 85 mm, pole height 31.5 mm, pole angle 22.5 deg
- RFD JLEIC 952.6 MHz: aperture 70 mm, c-c pole separation 170 mm(?)
- KEKB crab (arXiv 0706.3248, SRF2007 MO405): 508.9 MHz, kick 1.41/1.44 MV; squashed racetrack cell pushes other TM110 to ~700 MHz (above
  large pipe cutoff); coaxial coupler extracts TM010 at ~430 MHz (LOM) and TE; Esp/Vkick 14.4 (MV/m)/MV; ~21 MV/m
- 4-rod (Lancaster): two parallel sections of two longitudinally opposing lambda/4 rods; f set by rod length; transverse < 142 mm radius;
  [R/Q]T 915 Ohm, G 62.8 Ohm, Ep*=4.0 MV/m/MV, Bp*=7.56 mT/MV (NAPAC2016 slides); 3 MV: Ep 27.7 MV/m, Bp 63.9 mT (2011 sim)
- KEK cERL deflector (IPAC2021 MOPAB330): NC single cell TM110 pillbox, no nose cone, 5.2 GHz, 12 cells? ; IPAC10 MOPE002 2.6 GHz single cell deflector
## Proton linac cells
- Linac4 PIMS (PAC2009 FR5REP051; CERN-2020-006): 352.2 MHz, 102-160 MeV, 3.9 MV/m, 12 cav x 7 cells, k 5.6-5.0 %, ZTT 23.1-26 MOhm/m,
  Es 1.8 Kilp; gap length tunes frequency, cell inner radius constant; scaled LEP NC structure
- LIBO (arXiv 0712.0830): 2998 MHz SCL; web thickness 4 mm; cone angle ~20 deg optimum; ACLIP aperture radius 4 mm; 16 MV/m
- 700 MHz SCL (osti etdeweb 20146433): bore radii 2.00, 2.25, 2.50 cm
- arXiv 1804.08525 (Accelerating structures review): 3 GHz beta 0.4: -1 mm bore radius -> +10 % ZT^2; septum similar
- ACS J-PARC 972 MHz (from 1296 MHz design), 190-400 MeV; ACS 30 deg cone (arXiv 1306.6287)
- Superfish PAC93 0790: CCLFISH tunes diameter, septum, gap or cone angle; symmetric half cells
## Klystron
- PAC1993_1259 (PEP-II 476 MHz klystron): cavity R/Q ~100-108 Ohm, transit angles 0.95-1.14 rad; cavity 1: 476.3 MHz, R/Q 108, Qext 1000, theta 0.95
- IPAC2016 MOPMY014?: normalized drift tube radius 0.67, normalized beam radius 0.43 (SLAC design); output cavity small gap -> coupling
## Own computations (scratch scripts a3_axisym.py, a3_family_stats.py, a3_scan.py, a3_gun.py, a3_slac.py)
- axisym TM0 H_phi solver: pillbox R=d=5 cm: f err 3e-11, R/Q 222.82 vs 222.75 (3e-4), G 226.49 (5e-10)
- reentrant (400 draws, MIN_FILLET 0.05): R/lambda 0.216-0.359 (med 0.302); g/lambda 0.015-0.202 (med 0.074, p95 0.162);
  R/Q linac 74-321 (med 161, p95 255); G 74-200 (med 127, p95 174); T 0.87-0.99; rt/R 0.19-0.44; rn/R 0.15-0.60;
  rho_tip at floor 43 %; shares: R/Q>=226 13 %, >=290 0.5 %, G>=175 5 %, both(226,175) 4.3 %, g/lam<=0.1 71 %, g/lam>=0.2 0.3 %
  R/Q driven by Lc/R (+) and g/R (+), thin noses; G by g/R (+), nose length (-)
- pillbox_pipes (400): R/lambda 0.384-0.410; R/Q 91-205 (med 149); G 139-323 (med 247); T 0.39-0.93; rho_iris at floor 53 %
- scan beyond ranges (rp/R 0.15, tt 0.1R, alpha 20): Lc/R 1.2-1.4, g/Lc 0.4-0.6 -> R/Q 278-315, G 157-212, g/lam 0.14-0.28
  Lc/R 1.0-1.4, g/Lc 0.75 -> G 225-254 with R/Q 235-259
- MAX IV-like ratios (rt 0.146R, g 0.122R, rp 0.06R): R/lam 0.267 (Lc/R 0.8) ... 0.157 (Lc/R 2.0); G 120 -> 75
- gun rebuild (BNL/SLAC/UCLA table + assumed full-round iris t = 19 mm): 2859.1 / 2879.0 MHz (sep 19.9) vs 2852.5/2856.0 (3.5);
  iris round x0.8: 2843.7/2859.9; x1.2: 2875.0/2900.4 -> f within 0.1-0.7 %, separation not reproduced
- SLAC cavity 79 (a 9.95, b 40.96, t 5.84, d 34.99 mm; full-round iris assumed), 3 periods closed by E walls at mid-cells:
  0: 2837.5, pi/3: 2843.4, 2pi/3: 2855.2 (published 2856), pi: 2861.1 MHz
- G from published Q0 with ideal Cu (5.8e7): EU undamped 240, ESRF-EBS 175, PEP-II 176, ALS 167, KEK PF 258, MAX IV 55, LCLS gun 195, CLIC-G 158-164
- R/Q conventions (inferred): PEP-II Ks=V^2/2P -> 113 circ / 226 linac; EU 148.5 & ESRF 145 likely circuit (ALBA 700 kV, 80 kW,
  3.3 MOhm consistent with V^2/2P) -> ~290-300 linac; MAX IV linac 3.52e6/20923 = 168; KEK PF 111 = pipe-less pillbox optimum (circuit)
- evanescent decay at PEC caps (Lp = 1.5-4 rp): TM01 0.03-1e-4; TE11 for dipoles at rp/R 0.35: 0.15-6e-3; k R = 5 & rp/R 0.35 TE11: 0.42-0.10
- code: ridged_box QoI axis = default z-line through (0,0,0) = filleted box edge (box spans [0,a]x[0,b]x[0,d]); pipes are along x
