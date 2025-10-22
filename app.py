import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy import signal

# Initialize the Dash app
app = dash.Dash(__name__, title="Beat Visualizer Pro")
server = app.server  # For Vercel deployment

# Modern color scheme
COLORS = {
    'primary': '#00F5FF', 'secondary': '#FF0080', 'accent': '#FFD700',
    'success': '#00FFA3', 'wave1': '#A855F7', 'wave2': '#FF0080',
    'envelope': '#FFD700', 'bg_dark': '#0B0F1F', 'bg_light': '#1E2538',
    'bg_panel': '#252D42', 'text': '#F0F4FF', 'grid': '#2D3561'
}

# Preset configurations
PRESETS = {
    'musical': {'f1': 440, 'f2': 445, 'A': 1.0, 'damping': 0.0, 'name': '🎼 Musical'},
    'bass': {'f1': 120, 'f2': 125, 'A': 1.5, 'damping': 0.2, 'name': '🔊 Bass'},
    'treble': {'f1': 380, 'f2': 390, 'A': 1.0, 'damping': 0.1, 'name': '🎵 Treble'},
    'fast': {'f1': 300, 'f2': 320, 'A': 1.2, 'damping': 0.0, 'name': '⚡ Fast'},
    'slow': {'f1': 250, 'f2': 252, 'A': 1.0, 'damping': 0.0, 'name': '🌊 Slow'},
    'damped': {'f1': 260, 'f2': 250, 'A': 1.0, 'damping': 2.5, 'name': '💫 Damped'}
}

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('🎵 Ultimate Beat Phenomenon Simulator', 
                style={'textAlign': 'center', 'color': COLORS['primary'], 
                       'marginBottom': '10px', 'fontWeight': 'bold', 'fontSize': '2.5rem'}),
        html.P('Interactive Real-Time Wave Physics Visualization',
               style={'textAlign': 'center', 'color': COLORS['text'], 
                      'fontSize': '1.2rem', 'marginTop': '0'})
    ], style={'padding': '30px 20px', 'backgroundColor': COLORS['bg_dark'], 
              'borderBottom': f'3px solid {COLORS["primary"]}'}),
    
    # Main container
    html.Div([
        # Control Panel (Left Side - Scrollable)
        html.Div([
            html.H3('⚙️ Control Panel', style={'color': COLORS['accent'], 'marginBottom': '20px'}),
            
            # Preset Buttons
            html.Div([
                html.Label('Quick Presets:', style={'color': COLORS['text'], 'fontWeight': 'bold', 
                                                     'marginBottom': '10px', 'display': 'block'}),
                html.Div([
                    html.Button(preset['name'], id=f'preset-{key}', 
                               style={'margin': '5px', 'padding': '12px 24px', 
                                      'backgroundColor': COLORS['bg_panel'], 
                                      'color': COLORS['text'], 
                                      'border': f'2px solid {COLORS["primary"]}',
                                      'borderRadius': '10px', 'cursor': 'pointer', 
                                      'fontWeight': 'bold', 'fontSize': '14px',
                                      'transition': 'all 0.3s'})
                    for key, preset in PRESETS.items()
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '30px'})
            ]),
            
            # Frequency 1
            html.Div([
                html.Label('🎵 Frequency 1 (Hz):', style={'color': COLORS['wave1'], 
                                                           'fontWeight': 'bold', 'fontSize': '16px'}),
                dcc.Slider(id='freq1-slider', min=100, max=500, step=1, value=260,
                          marks={100: '100', 250: '250', 400: '400', 500: '500'},
                          tooltip={"placement": "bottom", "always_visible": True}),
                dcc.Input(id='freq1-input', type='number', value=260, min=100, max=500,
                         placeholder='Enter frequency...',
                         style={'width': '100%', 'padding': '12px', 'marginTop': '10px',
                                'backgroundColor': COLORS['bg_panel'], 'color': COLORS['text'],
                                'border': f'2px solid {COLORS["wave1"]}', 'borderRadius': '8px',
                                'fontSize': '16px'})
            ], style={'marginBottom': '30px'}),
            
            # Frequency 2
            html.Div([
                html.Label('🎶 Frequency 2 (Hz):', style={'color': COLORS['wave2'], 
                                                           'fontWeight': 'bold', 'fontSize': '16px'}),
                dcc.Slider(id='freq2-slider', min=100, max=500, step=1, value=250,
                          marks={100: '100', 250: '250', 400: '400', 500: '500'},
                          tooltip={"placement": "bottom", "always_visible": True}),
                dcc.Input(id='freq2-input', type='number', value=250, min=100, max=500,
                         placeholder='Enter frequency...',
                         style={'width': '100%', 'padding': '12px', 'marginTop': '10px',
                                'backgroundColor': COLORS['bg_panel'], 'color': COLORS['text'],
                                'border': f'2px solid {COLORS["wave2"]}', 'borderRadius': '8px',
                                'fontSize': '16px'})
            ], style={'marginBottom': '30px'}),
            
            # Amplitude
            html.Div([
                html.Label('📊 Amplitude:', style={'color': COLORS['accent'], 
                                                   'fontWeight': 'bold', 'fontSize': '16px'}),
                dcc.Slider(id='amp-slider', min=0.1, max=2.0, step=0.1, value=1.0,
                          marks={0.1: '0.1', 1.0: '1.0', 2.0: '2.0'},
                          tooltip={"placement": "bottom", "always_visible": True}),
                dcc.Input(id='amp-input', type='number', value=1.0, min=0.1, max=2.0, step=0.1,
                         placeholder='Enter amplitude...',
                         style={'width': '100%', 'padding': '12px', 'marginTop': '10px',
                                'backgroundColor': COLORS['bg_panel'], 'color': COLORS['text'],
                                'border': f'2px solid {COLORS["accent"]}', 'borderRadius': '8px',
                                'fontSize': '16px'})
            ], style={'marginBottom': '30px'}),
            
            # Damping
            html.Div([
                html.Label('💨 Damping:', style={'color': COLORS['success'], 
                                                 'fontWeight': 'bold', 'fontSize': '16px'}),
                dcc.Slider(id='damp-slider', min=0.0, max=5.0, step=0.1, value=0.0,
                          marks={0: '0', 2.5: '2.5', 5: '5'},
                          tooltip={"placement": "bottom", "always_visible": True}),
                dcc.Input(id='damp-input', type='number', value=0.0, min=0.0, max=5.0, step=0.1,
                         placeholder='Enter damping...',
                         style={'width': '100%', 'padding': '12px', 'marginTop': '10px',
                                'backgroundColor': COLORS['bg_panel'], 'color': COLORS['text'],
                                'border': f'2px solid {COLORS["success"]}', 'borderRadius': '8px',
                                'fontSize': '16px'})
            ], style={'marginBottom': '30px'}),
            
            # Info Panel
            html.Div(id='info-panel', style={
                'padding': '20px', 'backgroundColor': COLORS['bg_panel'],
                'borderRadius': '12px', 'border': f'3px solid {COLORS["primary"]}',
                'marginTop': '20px', 'boxShadow': '0 4px 12px rgba(0,245,255,0.2)'
            })
            
        ], style={'width': '100%', 'maxWidth': '400px', 'padding': '25px', 
                  'backgroundColor': COLORS['bg_light'],
                  'borderRadius': '15px', 'margin': '20px', 
                  'boxShadow': '0 8px 16px rgba(0,0,0,0.4)',
                  'overflowY': 'auto', 'maxHeight': 'calc(100vh - 200px)',
                  'position': 'sticky', 'top': '20px'}),
        
        # Visualization Panel (Right Side - Scrollable)
        html.Div([
            dcc.Graph(id='main-graph', style={'marginBottom': '30px'}),
            dcc.Graph(id='spectrum-graph', style={'marginBottom': '30px'}),
            dcc.Graph(id='phase-graph', style={'marginBottom': '30px'}),
            dcc.Graph(id='heatmap-graph', style={'marginBottom': '30px'})
        ], style={'flex': '1', 'padding': '20px', 'overflowY': 'auto'})
        
    ], style={'display': 'flex', 'backgroundColor': COLORS['bg_dark'], 
              'minHeight': 'calc(100vh - 200px)', 'flexWrap': 'wrap'})
    
], style={'fontFamily': 'system-ui, -apple-system, sans-serif', 
          'backgroundColor': COLORS['bg_dark'], 'minHeight': '100vh'})

# Callback for syncing sliders and inputs
@app.callback(
    [Output('freq1-slider', 'value'), Output('freq1-input', 'value'),
     Output('freq2-slider', 'value'), Output('freq2-input', 'value'),
     Output('amp-slider', 'value'), Output('amp-input', 'value'),
     Output('damp-slider', 'value'), Output('damp-input', 'value')],
    [Input('freq1-slider', 'value'), Input('freq1-input', 'value'),
     Input('freq2-slider', 'value'), Input('freq2-input', 'value'),
     Input('amp-slider', 'value'), Input('amp-input', 'value'),
     Input('damp-slider', 'value'), Input('damp-input', 'value')] +
    [Input(f'preset-{key}', 'n_clicks') for key in PRESETS.keys()],
    prevent_initial_call=True
)
def sync_inputs(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle preset buttons
    if trigger_id.startswith('preset-'):
        preset_key = trigger_id.replace('preset-', '')
        preset = PRESETS[preset_key]
        return (preset['f1'], preset['f1'], preset['f2'], preset['f2'],
                preset['A'], preset['A'], preset['damping'], preset['damping'])
    
    # Sync slider and input
    f1_s, f1_i, f2_s, f2_i, a_s, a_i, d_s, d_i = args[:8]
    
    if trigger_id == 'freq1-slider':
        return f1_s, f1_s, f2_s, f2_i, a_s, a_i, d_s, d_i
    elif trigger_id == 'freq1-input':
        return f1_i, f1_i, f2_s, f2_i, a_s, a_i, d_s, d_i
    elif trigger_id == 'freq2-slider':
        return f1_s, f1_i, f2_s, f2_s, a_s, a_i, d_s, d_i
    elif trigger_id == 'freq2-input':
        return f1_s, f1_i, f2_i, f2_i, a_s, a_i, d_s, d_i
    elif trigger_id == 'amp-slider':
        return f1_s, f1_i, f2_s, f2_i, a_s, a_s, d_s, d_i
    elif trigger_id == 'amp-input':
        return f1_s, f1_i, f2_s, f2_i, a_i, a_i, d_s, d_i
    elif trigger_id == 'damp-slider':
        return f1_s, f1_i, f2_s, f2_i, a_s, a_i, d_s, d_s
    elif trigger_id == 'damp-input':
        return f1_s, f1_i, f2_s, f2_i, a_s, a_i, d_i, d_i
    
    return dash.no_update

# Main visualization callback
@app.callback(
    [Output('main-graph', 'figure'),
     Output('spectrum-graph', 'figure'),
     Output('phase-graph', 'figure'),
     Output('heatmap-graph', 'figure'),
     Output('info-panel', 'children')],
    [Input('freq1-slider', 'value'),
     Input('freq2-slider', 'value'),
     Input('amp-slider', 'value'),
     Input('damp-slider', 'value')]
)
def update_graphs(f1, f2, A, damping):
    # Calculate waves
    beat_freq = abs(f1 - f2) or 0.1
    beat_period = 1 / beat_freq
    avg_freq = (f1 + f2) / 2
    sample_rate = 2000
    
    t_end = max(2, min(5, int(2.0 * beat_freq))) * beat_period
    t = np.linspace(0, t_end, sample_rate)
    
    y1 = A * np.sin(2 * np.pi * f1 * t)
    y2 = A * np.sin(2 * np.pi * f2 * t)
    
    if damping > 0:
        damp_env = np.exp(-damping * t)
        y1, y2 = y1 * damp_env, y2 * damp_env
    
    y_resultant = y1 + y2
    envelope_upper = 2 * A * np.abs(np.cos(2 * np.pi * beat_freq / 2 * t))
    if damping > 0:
        envelope_upper *= np.exp(-damping * t)
    envelope_lower = -envelope_upper
    
    # Main waveform graph
    main_fig = go.Figure()
    main_fig.add_trace(go.Scatter(x=t, y=envelope_upper, mode='lines', name='Envelope',
                                  line=dict(color=COLORS['envelope'], width=2, dash='dash')))
    main_fig.add_trace(go.Scatter(x=t, y=envelope_lower, mode='lines', name='Envelope',
                                  line=dict(color=COLORS['envelope'], width=2, dash='dash'),
                                  showlegend=False))
    main_fig.add_trace(go.Scatter(x=t, y=y_resultant, mode='lines', name='Resultant Wave',
                                  line=dict(color=COLORS['primary'], width=3)))
    main_fig.add_trace(go.Scatter(x=t, y=y1, mode='lines', name=f'Wave 1 ({f1} Hz)',
                                  line=dict(color=COLORS['wave1'], width=2), visible='legendonly'))
    main_fig.add_trace(go.Scatter(x=t, y=y2, mode='lines', name=f'Wave 2 ({f2} Hz)',
                                  line=dict(color=COLORS['wave2'], width=2), visible='legendonly'))
    
    main_fig.update_layout(
        title=dict(text=f'🎵 Beat Phenomenon | Beat Frequency: {beat_freq:.2f} Hz',
                   font=dict(size=20, color=COLORS['primary'])),
        xaxis_title='Time (s)', yaxis_title='Amplitude',
        plot_bgcolor=COLORS['bg_light'], paper_bgcolor=COLORS['bg_panel'],
        font=dict(color=COLORS['text'], size=14), hovermode='x unified',
        legend=dict(bgcolor=COLORS['bg_panel'], bordercolor=COLORS['primary'], borderwidth=2),
        height=500
    )
    
    # Spectrum graph
    fft_vals = np.fft.fft(y_resultant)
    fft_freq = np.fft.fftfreq(len(y_resultant), 1/sample_rate)
    pos_mask = fft_freq > 0
    fft_freq_pos = fft_freq[pos_mask]
    fft_mag = np.abs(fft_vals[pos_mask]) / len(y_resultant)
    
    spectrum_fig = go.Figure()
    spectrum_fig.add_trace(go.Bar(x=fft_freq_pos, y=fft_mag, name='Magnitude',
                                  marker=dict(color=COLORS['primary'])))
    spectrum_fig.update_layout(
        title=dict(text='📊 Frequency Spectrum (FFT)', font=dict(size=18, color=COLORS['accent'])),
        xaxis_title='Frequency (Hz)', yaxis_title='Magnitude',
        plot_bgcolor=COLORS['bg_light'], paper_bgcolor=COLORS['bg_panel'],
        font=dict(color=COLORS['text'], size=14), xaxis_range=[0, max(f1, f2) + 100],
        height=400
    )
    
    # Phase space graph
    phase_fig = go.Figure()
    phase_fig.add_trace(go.Scatter(x=y1, y=y2, mode='lines', name='Lissajous',
                                   line=dict(color=COLORS['success'], width=2)))
    phase_fig.update_layout(
        title=dict(text='🌀 Phase Space (Lissajous Curve)', font=dict(size=18, color=COLORS['accent'])),
        xaxis_title='Wave 1', yaxis_title='Wave 2',
        plot_bgcolor=COLORS['bg_light'], paper_bgcolor=COLORS['bg_panel'],
        font=dict(color=COLORS['text'], size=14),
        height=400
    )
    
    # Spectrogram
    f_spec, t_spec, Sxx = signal.spectrogram(y_resultant, fs=sample_rate, nperseg=256, noverlap=200)
    
    heatmap_fig = go.Figure()
    heatmap_fig.add_trace(go.Heatmap(z=10 * np.log10(Sxx + 1e-10), x=t_spec, y=f_spec,
                                     colorscale='Magma', name='Power'))
    heatmap_fig.update_layout(
        title=dict(text='🔥 Time-Frequency Spectrogram', font=dict(size=18, color=COLORS['accent'])),
        xaxis_title='Time (s)', yaxis_title='Frequency (Hz)',
        plot_bgcolor=COLORS['bg_light'], paper_bgcolor=COLORS['bg_panel'],
        font=dict(color=COLORS['text'], size=14), yaxis_range=[0, max(f1, f2) + 100],
        height=400
    )
    
    # Info panel
    info = html.Div([
        html.H4('📊 Simulation Parameters', style={'color': COLORS['primary'], 
                                                    'marginBottom': '15px', 'fontSize': '18px'}),
        html.Div([
            html.P([html.Strong('📐 Wave Properties:', style={'color': COLORS['accent']}),
                   html.Br(), f'• Frequency 1: {f1:.1f} Hz',
                   html.Br(), f'• Frequency 2: {f2:.1f} Hz',
                   html.Br(), f'• Amplitude: {A:.2f}',
                   html.Br(), f'• Damping: {damping:.2f}'],
                  style={'color': COLORS['text'], 'marginBottom': '15px', 'fontSize': '14px'}),
            html.P([html.Strong('🎵 Beat Analysis:', style={'color': COLORS['accent']}),
                   html.Br(), f'• Beat Frequency: {beat_freq:.3f} Hz',
                   html.Br(), f'• Beat Period: {beat_period:.4f} s',
                   html.Br(), f'• Average Frequency: {avg_freq:.1f} Hz',
                   html.Br(), f'• Δf: {abs(f1-f2):.1f} Hz'],
                  style={'color': COLORS['text'], 'marginBottom': '15px', 'fontSize': '14px'}),
            html.P([html.Strong('⚡ Physics:', style={'color': COLORS['accent']}),
                   html.Br(), '• Superposition: ✓',
                   html.Br(), '• Interference: ✓',
                   html.Br(), f'• Damping: {"✓" if damping > 0 else "✗"}'],
                  style={'color': COLORS['text'], 'fontSize': '14px'})
        ])
    ])
    
    return main_fig, spectrum_fig, phase_fig, heatmap_fig, info

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)