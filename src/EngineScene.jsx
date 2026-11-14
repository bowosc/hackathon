// The shared hand and fruit keep the interaction consistent across the pipeline.
function Interaction({ stage, banana = false }) {
  return <g strokeLinecap="round" strokeLinejoin="round">
    <ellipse cx="143" cy="139" rx="49" ry="6" fill="#35452d" opacity=".08" />
    {banana ? <>
      <path d="M171 44c13 31 1 65-27 80-18 10-37 7-49-7 37 0 61-30 68-65Z" fill={`url(#${stage}-fruit)`} stroke="#b59b3d" strokeWidth="1" />
      <path d="M108 119c25-6 48-31 57-65" fill="none" stroke="#fff5bf" strokeWidth="2" />
      <path d="m165 52 5-10 5 1-3 10" fill="#788051" />
    </> : <>
      <path d="M156 58c-16-13-37-3-37 19-2 23 12 47 26 48 6 1 9-3 15-1 16 7 33-22 34-43 2-24-21-38-38-23Z" fill={`url(#${stage}-fruit)`} stroke="#c0a64e" strokeWidth=".9" />
      <ellipse cx="136" cy="78" rx="7" ry="13" transform="rotate(22 136 78)" fill="#fff8d8" opacity=".4" />
      <path d="M156 58c-2-9 1-15 5-20" stroke="#716645" strokeWidth="2.8" />
      <path d="M159 49c0-12 12-18 23-13-3 12-14 18-23 13Z" fill="#889773" /><path d="m160 48 15-9" stroke="#bdc9a7" strokeWidth="1" />
    </>}
    <path d="M12 105h60c11 0 15-13 25-20l20-13c7-4 13 2 9 8l-15 14 24-6c8-2 12 6 5 10l-16 9 19-3c8-1 11 8 3 11l-19 6h13c8 0 10 7 2 10l-29 6c-14 3-27 3-39-3H12Z" fill={`url(#${stage}-skin)`} stroke="#b39d80" strokeWidth="1" />
    <path d="m112 96-12 9m25 2-17 5m19 9-17 2" stroke="#baa286" strokeWidth="1.1" fill="none" />
    <path d="M12 103h51l4 34H12Z" fill="#dce3d3" stroke="#a3af95" strokeWidth="1" /><path d="m58 104 4 31" stroke="#f8faf1" strokeWidth="2" />
  </g>;
}

export default function EngineScene({ stage }) {
  return <svg className={`engine-scene scene-${stage}`} viewBox="0 0 240 185" fill="none" aria-hidden="true">
    <defs>
      <radialGradient id={`${stage}-fruit`} cx=".32" cy=".25" r=".85"><stop stopColor="#f9e79c" /><stop offset=".6" stopColor="#edcf67" /><stop offset="1" stopColor="#d0ac3f" /></radialGradient>
      <linearGradient id={`${stage}-skin`} x1="20" y1="80" x2="110" y2="140" gradientUnits="userSpaceOnUse"><stop stopColor="#edddc5" /><stop offset="1" stopColor="#d3b998" /></linearGradient>
      <linearGradient id={`${stage}-canvas`} x1="0" y1="0" x2="1" y2="1"><stop stopColor="#fbfcf8" /><stop offset="1" stopColor="#ecefe4" /></linearGradient>
      <clipPath id={`${stage}-frame`}><rect x="9" y="13" width="222" height="155" rx="10" /></clipPath>
      <clipPath id={`${stage}-tile`}><rect width="67" height="119" rx="7" /></clipPath>
    </defs>
    {stage === 'segment' && <>
      <rect x="9" y="13" width="222" height="155" rx="10" fill="#f3f4eb" stroke="#d8dfcc" />
      <g clipPath={`url(#${stage}-frame)`}>
        <path d="M9 116 120 86l111 33v49H9Z" fill="#e7ebdc" />
        <path d="m120 91-38 77m38-77 54 77M9 143h222" stroke="#d6deca" />
        {[{x:23,y:49,width:57},{x:160,y:47,width:56}].map(shelf => <g key={shelf.x}>
          <rect x={shelf.x} y={shelf.y} width={shelf.width} height="78" rx="3" fill="#e4e8d9" stroke="#bac6ac" />
          {[0,1,2].map(row => <g key={row}>
            {[0,1,2,3].map(column => <rect key={column} x={shelf.x+7+column*12} y={shelf.y+8+row*23} width="8" height={column % 2 ? 15 : 13} rx="1.5" fill={['#c0cdae','#e8d586','#d2cfb0','#b0bea0'][(column+row)%4]} stroke="#aab79a" strokeWidth=".5" />)}
            <path d={`M${shelf.x+1} ${shelf.y+27+row*23}h${shelf.width-2}`} stroke="#b8c5aa" />
          </g>)}
        </g>)}
        <ellipse cx="118" cy="140" rx="23" ry="4" fill="#a4b195" opacity=".25" />
        <path d="m108 104 11 1-9 32h-11Zm10 1 8-1 6 32h-10Z" fill="#607252" />
        <path d="M99 134h12v5H95c-2-2 0-4 4-5m23 0h10l6 4c1 2-1 2-3 2h-13Z" fill="#48563d" />
        <path d="M105 78c6-6 15-7 21-1l4 30c-8 5-19 5-27 0Z" fill="#8fa17b" stroke="#758964" strokeWidth="1" />
        <path d="m108 83-10 20" stroke="#8fa17b" strokeWidth="8" strokeLinecap="round" /><path d="m98 103-3 6" stroke="#d1b695" strokeWidth="5" strokeLinecap="round" />
        <path d="m124 82 15 9 15-9" stroke="#8fa17b" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" /><path d="m152 83 8-3" stroke="#d1b695" strokeWidth="5" strokeLinecap="round" />
        <rect x="111" y="66" width="9" height="12" rx="4" fill="#d5bd9d" />
        <ellipse cx="115" cy="61" rx="10" ry="12" fill="#e5d1b5" />
        <path d="M105 62c-4-12 5-19 13-15 7 0 11 7 8 12l-5-5c-4 6-10 3-16 8Z" fill="#697555" />
        <path d="M166 77c-5-4-10 0-9 5 1 6 5 12 9 10 5 2 9-6 10-11 0-6-7-8-10-4Z" fill="#edcf70" stroke="#bba54f" strokeWidth=".8" /><path d="m166 76 2-5" stroke="#71824f" strokeWidth="1.5" />
      </g>
      <circle cx="23" cy="25" r="2.5" fill="#9aaa7f" /><text x="32" y="28" className="diagram-micro">SOURCE FOOTAGE</text>
      <rect x="93" y="42" width="90" height="103" rx="5" stroke="#8a9d72" strokeDasharray="3 4" />
      <rect x="147" y="68" width="34" height="31" rx="3" fill="#e8d36c" fillOpacity=".1" stroke="#b9a343" strokeWidth="1.5" />
      <path d="m164 99 16 15" stroke="#b9a343" strokeWidth="1.2" />
      <rect x="156" y="114" width="67" height="46" rx="5" fill="#fff9e7" stroke="#d6c16c" />
      <g transform="translate(158 113) scale(.31)"><Interaction stage={stage} /></g>
    </>}
    {stage === 'refine' && <>
      <rect x="9" y="13" width="222" height="155" rx="10" fill={`url(#${stage}-canvas)`} stroke="#dce1d2" />
      <rect x="18" y="35" width="93" height="95" rx="7" fill="#e8ecdf" /><rect x="129" y="35" width="93" height="95" rx="7" fill="#fffdf4" stroke="#e1e5d8" />
      <g transform="translate(24 48)">
        {["         ll  ", "        lll  ", "       yyyyy ", "       yyyyyy", "      hyyyyyy", "    hhhyyyyyy", "sssshhhhyyyy ", "sssshhhhyyyy ", "sssshhhhhyy  ", "sssshhhh     ", "sssshhh      "].map((row, y) => [...row].map((pixel, x) => pixel !== ' ' && <rect key={`${x}-${y}`} x={x * 6} y={y * 6} width="6" height="6" fill={{ l: '#8b9a72', y: '#e8cb70', h: '#ddc7a6', s: '#cad5bd' }[pixel]} />))}
      </g>
      <path d="M115 83h10m-4-4 4 4-4 4" stroke="#8b9979" strokeWidth="1.3" strokeLinecap="round" />
      <g transform="translate(128 40) scale(.47)"><Interaction stage={stage} /></g>
      <path d="m206 42 2-6 2 6 6 2-6 2-2 6-2-6-6-2Z" fill="#dfc45c" />
      <text x="64" y="150" textAnchor="middle" className="diagram-micro">SOURCE</text><text x="175" y="150" textAnchor="middle" className="diagram-micro">REFINED</text>
    </>}
    {stage === 'label' && <>
      <rect x="9" y="13" width="222" height="155" rx="10" fill={`url(#${stage}-canvas)`} stroke="#dce1d2" />
      <g transform="translate(4 -5) scale(1.03)"><Interaction stage={stage} /></g>
      <rect x="120" y="43" width="86" height="86" rx="4" stroke="#b9a24e" strokeWidth="1" />
      <rect x="74" y="72" width="79" height="67" rx="4" stroke="#8c9f7a" strokeWidth="1" />
      <rect x="120" y="30" width="42" height="15" rx="3" fill="#eee0a0" /><text x="127" y="40" className="diagram-label">apple</text>
      <rect x="74" y="59" width="39" height="15" rx="3" fill="#d4dfc8" /><text x="81" y="69" className="diagram-label">hand</text>
      <path d="m91 117 18-14 21 6 13 13" stroke="#6b8556" strokeWidth="1.2" />
      <g fill="#6b8556" stroke="#f3f7ec" strokeWidth="1">{[[91,117],[109,103],[130,109],[143,122]].map(([cx,cy]) => <circle key={cx} cx={cx} cy={cy} r="2.6" />)}</g>
      <text x="120" y="158" textAnchor="middle" className="diagram-action">reach → grasp → place</text>
    </>}
    {stage === 'multiply' && <>
      {[{x:9,fill:'#faf4db'}, {x:87,fill:'#414f37',banana:true}, {x:165,fill:'#e2e9d9'}].map((variant, index) => <g key={variant.x} transform={`translate(${variant.x} 30)`}>
        <rect width="67" height="119" rx="7" fill={variant.fill} stroke={index === 1 ? '#414f37' : '#d5ddc8'} />
        <g clipPath={`url(#${stage}-tile)`}>
          {index === 2 && <path d="M36 0h31v119H0V75Z" fill="#aabf93" opacity=".24" />}
          <g transform="translate(-3 24) scale(.36)"><Interaction stage={stage} banana={variant.banana} /></g>
          <path d="M11 99h18m-18 5h31" stroke={index === 1 ? '#abb79b' : '#bcc6ac'} strokeWidth="2" strokeLinecap="round" />
        </g>
        <circle cx="54" cy="12" r="2.5" fill={index === 1 ? '#e6d17b' : '#b1bf9c'} />
      </g>)}
      <text x="120" y="170" textAnchor="middle" className="diagram-action">objects · lighting · backgrounds</text>
    </>}
  </svg>;
}
