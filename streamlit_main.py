import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import odeformer # Assuming odeformer is installed and accessible
from odeformer.model import SymbolicTransformerRegressor
from odeformer.metrics import r2_score

# --- Configuration ---
DEFAULT_SEQ_LENGTH = 50
DEFAULT_T_START = 1.0
DEFAULT_T_END = 10.0
DEC_LAYERS = 12 # Assuming the model always has 12 decoder layers
TRANS_HEADS = 16 # Assuming the model always has 16 heads
MODEL_ARGS = {'beam_size': 2, 'beam_temperature': 0.1}
BEAM_IDX_TO_PLOT = 0 # Select which beam's attention to visualize

# Set seed for reproducibility (optional but good practice)
np.random.seed(2)
# torch.manual_seed(2) # Add if model has torch-based randomness

# --- Cached Model Loading ---
# Use st.cache_resource for objects that should persist across sessions/reruns
# and are expensive to create (like loading a model).
@st.cache_resource
def load_model():
    print("Loading ODEformer model...") # Add print statement to see when it loads
    # Ensure store_attentions=True is set here if needed globally
    dstr = SymbolicTransformerRegressor(from_pretrained=True, plot_token_charts=False, store_attentions=True)
    dstr.set_model_args(MODEL_ARGS)
    # Specific settings if they are fixed:
    dstr.model.encoder.ignore_enc_layers = []
    dstr.model.decoder.k_to_store = 0 # Check if this impacts intermediate_tokens storage
    print("Model loaded.")
    return dstr

# --- Plotting Functions (Modified for Streamlit) ---

def plot_input_trajectory(times, trajectory):
    """Plots the input trajectory points."""
    fig, ax = plt.subplots(figsize=(6, 3)) # Adjusted size
    dimension = trajectory.shape[1]
    for dim in range(dimension):
        ax.scatter(times, trajectory[:, dim], label=f'Input $x_{dim}$', marker='o', alpha=0.6)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Input Trajectory")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_predicted_trajectory(times, trajectory, pred_trajectory):
    """Plots the input points vs the model's predicted trajectory."""
    fig, ax = plt.subplots(figsize=(6, 3)) # Adjusted size
    dimension = trajectory.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, dimension))
    for dim in range(dimension):
        ax.scatter(times, trajectory[:, dim], color=colors[dim], label=f'Input $x_{dim}$', marker='o', alpha=0.3)
        ax.plot(times, pred_trajectory[:, dim], color=colors[dim], label=f'Predicted $x_{dim}$')
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Input vs. Predicted Trajectory")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

# def plot_single_cross_attention(decoder_cross_attn_layer_beam, times, ytick_labels, layer_idx, seq_length):
#     """Plots cross-attention for a single decoder layer and beam."""
#     fig = plt.figure(figsize=(16, 12)) # Adjusted size
#     gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 1])
#
#     # Plot settings
#     # Reduce frequency of x-axis labels if sequence length is large
#     tick_step = max(1, seq_length // 20)
#     xtick_indices = np.arange(0, seq_length, tick_step)
#     xtick_labels_full = [f"t={t:.1f}" for t in times] # Simpler labels for space
#     xtick_labels_display = [xtick_labels_full[i] for i in xtick_indices]
#
#     yticks = np.arange(len(ytick_labels))
#
#     # Plot 1: Sum of all heads
#     ax1 = fig.add_subplot(gs[0, :])
#     sum_attn = decoder_cross_attn_layer_beam.sum(dim=0).numpy()
#     if sum_attn.max() > 0:
#         im = ax1.imshow(sum_attn, cmap='viridis', aspect='auto', interpolation='nearest')
#         plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.01) # Reduced padding
#     else:
#         ax1.imshow(np.zeros_like(sum_attn), cmap='viridis', aspect='auto', vmin=0, vmax=1) # Show empty plot if no attn
#         st.warning(f"Layer {layer_idx}, Beam {BEAM_IDX_TO_PLOT}: Sum attention is zero.")
#
#
#     ax1.set_title(f"Decoder Layer {layer_idx} Beam {BEAM_IDX_TO_PLOT} - Sum of All Heads", pad=10) # Reduced padding
#     ax1.set_xticks(xtick_indices)
#     ax1.set_xticklabels(xtick_labels_display, rotation=45, ha='right', fontsize=8) # Smaller font
#     ax1.set_yticks(yticks)
#     ax1.set_yticklabels(ytick_labels, fontsize=8) # Smaller font
#
#     # Plot 2-17: Individual heads
#     for head_idx in range(TRANS_HEADS):
#         row = (head_idx // 4) + 1
#         col = head_idx % 4
#         ax = fig.add_subplot(gs[row, col])
#
#         head_attn = decoder_cross_attn_layer_beam[head_idx].numpy()
#         if head_attn.max() > 0:
#             im_head = ax.imshow(head_attn, cmap='viridis', aspect='auto', interpolation='nearest')
#             # plt.colorbar(im_head, ax=ax, fraction=0.046, pad=0.04) # Optional: colorbar per head
#         else:
#              ax.imshow(np.zeros_like(head_attn), cmap='viridis', aspect='auto', vmin=0, vmax=1) # Show empty plot if no attn
#
#         ax.set_title(f"Head {head_idx}", fontsize=9) # Smaller font
#         ax.set_xticks(xtick_indices[::2]) # Even fewer labels for small plots
#         ax.set_xticklabels(xtick_labels_display[::2], rotation=45, ha='right', fontsize=7) # Smaller font
#         ax.set_yticks(yticks[::2]) # Fewer labels
#         ax.set_yticklabels(ytick_labels[::2], fontsize=7) # Smaller font
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
#     return fig


def plot_cross_attention_interactive(decoder_cross_attn_all_layers, decoder_scaled_cross_attn_all_layers,
                                     # times, ytick_labels, seq_length, available_layers):
                                     times, seq_length, available_layers):
    """Creates an interactive Plotly figure that can switch between layers and attention types"""

    # Create subplots layout
    fig = make_subplots(
        rows=5, cols=4,
        specs=[[{"colspan": 4}] + [None] * 3] + [[{} for _ in range(4)] for _ in range(4)],
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],
        vertical_spacing=0.05
    )

    # Simplify time labels
    tick_step = max(1, seq_length // 20)
    xtick_indices = list(range(0, seq_length, tick_step))
    xtick_labels = [f"t={times[i]:.1f}" for i in xtick_indices]

    # Add traces for all combinations of layers and attention types
    # Initially all traces are hidden except for layer 0, regular attention

    # Create a lookup table for trace visibility state
    layer_visibility_states = {}

    for layer_idx in available_layers:
        # For each layer, we'll have two visibility states: regular and scaled
        trace_indices = {}  # Store indices for this layer

        decoder_cross_attn_regular = decoder_cross_attn_all_layers[layer_idx]
        decoder_cross_attn_scaled = decoder_scaled_cross_attn_all_layers[layer_idx]

        # Sum trace for regular attention
        sum_attn_reg = np.flipud(decoder_cross_attn_regular.sum(dim=0).numpy())
        idx = len(fig.data)
        trace_indices['sum_reg'] = idx
        fig.add_trace(go.Heatmap(
            z=sum_attn_reg, colorscale='Viridis',
            # x=list(range(seq_length)), y=ytick_labels,
            x=list(range(seq_length)),
            visible=(layer_idx == available_layers[0]),  # First layer visible initially
            name=f'L{layer_idx}-Regular-Sum'
        ), row=1, col=1)

        # Sum trace for scaled attention
        sum_attn_scaled = np.flipud(decoder_cross_attn_scaled.sum(dim=0).numpy())
        idx = len(fig.data)
        trace_indices['sum_scaled'] = idx
        fig.add_trace(go.Heatmap(
            z=sum_attn_scaled, colorscale='Viridis',
#             x=list(range(seq_length)), y=ytick_labels,
            x=list(range(seq_length)),
            visible=False,  # Hidden initially
            name=f'L{layer_idx}-Scaled-Sum'
        ), row=1, col=1)

        # Individual head traces
        for head_idx in range(16):  # Assuming 16 heads
            row = (head_idx // 4) + 2
            col = (head_idx % 4) + 1

            # Regular attention for this head
            head_attn_reg = np.flipud(decoder_cross_attn_regular[head_idx].numpy())
            idx = len(fig.data)
            trace_indices[f'head_{head_idx}_reg'] = idx
            fig.add_trace(go.Heatmap(
                z=head_attn_reg, colorscale='Viridis',
                showscale=False,  # Hide individual colorbars
                # x=list(range(seq_length)), y=ytick_labels,
                x=list(range(seq_length)),
                visible=(layer_idx == available_layers[0]),  # First layer visible initially
                name=f'L{layer_idx}-Regular-Head{head_idx}'
            ), row=row, col=col)

            # Scaled attention for this head
            head_attn_scaled = np.flipud(decoder_cross_attn_scaled[head_idx].numpy())
            idx = len(fig.data)
            trace_indices[f'head_{head_idx}_scaled'] = idx
            fig.add_trace(go.Heatmap(
                z=head_attn_scaled, colorscale='Viridis',
                showscale=False,  # Hide individual colorbars
                # x=list(range(seq_length)), y=ytick_labels,
                x=list(range(seq_length)),
                visible=False,  # Hidden initially
                name=f'L{layer_idx}-Scaled-Head{head_idx}'
            ), row=row, col=col)

        # Store trace indices for this layer
        layer_visibility_states[layer_idx] = trace_indices

    # Create buttons for toggling between regular and scaled attention
    attention_buttons = []

    # For each layer, create complete visibility arrays
    for layer_idx in available_layers:
        # Regular attention for this layer (make only regular traces visible for this layer)
        reg_visibility = []
        for i in range(len(fig.data)):
            trace_layer = i // 34  # Each layer has 34 traces
            is_current_layer = (trace_layer == layer_idx)
            is_regular = (i % 2 == 0)
            visible = is_current_layer and is_regular
            reg_visibility.append(visible)

        # Scaled attention for this layer (make only scaled traces visible for this layer)
        scaled_visibility = []
        for i in range(len(fig.data)):
            trace_layer = i // 34
            is_current_layer = (trace_layer == layer_idx)
            is_regular = (i % 2 == 0)
            visible = is_current_layer and not is_regular
            scaled_visibility.append(visible)

        # Button for regular attention will set the exact visibility array for this layer
        attention_buttons.append(dict(
            method="update",
            label=f"Layer {layer_idx}: Regular",
            args=[{"visible": reg_visibility}]
        ))

        # Button for scaled attention will set the exact visibility array for this layer
        attention_buttons.append(dict(
            method="update",
            label=f"Layer {layer_idx}: Scaled",
            args=[{"visible": scaled_visibility}]
        ))

    # Update layout with a single updatemenus for all combinations
    fig.update_layout(
        height=900,
        title="Decoder Attention Visualization",
        updatemenus=[dict(
            buttons=attention_buttons,
            direction="down",
            showactive=True,
            x=0.5,
            y=1.15,
            xanchor="center",
            yanchor="top"
        )]
    )

    # Fix the visibility logic for layer and attention type toggles
    for i, trace in enumerate(fig.data):
        layer_idx = i // (34)  # 34 traces per layer (1 sum + 16 heads) * 2 (reg/scaled)
        is_regular = (i % 2 == 0)

        # Set initial visibility: only show layer 0 regular traces
        if layer_idx == 0 and is_regular:
            trace.visible = True
        else:
            trace.visible = False

    # Update x-axis labels
    for i in range(1, 18):  # All subplots
        fig.update_xaxes(tickvals=xtick_indices, ticktext=xtick_labels, tickangle=45)

    return fig


def plot_single_token_chart(intermediate_tokens_layer_data, dstr, layer_idx):
    """Plots the token chart for a single decoder layer."""
    if not intermediate_tokens_layer_data:
         st.warning(f"No intermediate token data available for layer {layer_idx}.")
         return None # Return None if no data

    # Assuming structure is [ (token_pos, [beam0_token, beam1_token,...]), ... ]
    # Or potentially just [ [beam0_token, beam1_token,...], ... ] if token_pos isn't the first element
    # Let's assume it's like the original code's expectation: list of tuples/lists where the second element holds beams
    try:
        # Extract beam data - adjust based on actual structure of intermediate_tokens_layer_data
        beam_data = [item[1] for item in intermediate_tokens_layer_data] # Original assumption
        num_beams = len(beam_data[0]) if beam_data else 0
        num_token_steps = len(beam_data)

        if num_beams == 0 or num_token_steps == 0:
            st.warning(f"Token data seems empty for layer {layer_idx}.")
            return None

        row_headers = [f"Token {i}" for i in range(num_token_steps)]
        col_headers = [f"Beam {i}" for i in range(num_beams)]

        id2word_size = dstr.model.decoder.n_words
        norm = mcolors.Normalize(vmin=0, vmax=id2word_size - 1)
        # Ensure 'tab20' has enough colors or choose a different map if id2word_size > 20
        cmap = plt.cm.get_cmap('tab20', id2word_size)

        fig, ax = plt.subplots(figsize=(max(10, num_beams * 1.5), max(3, num_token_steps * 0.5))) # Dynamic size
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data and colors (transpose of original logic for row/col headers)
        table_data = [[beam_data[step][beam] for beam in range(num_beams)] for step in range(num_token_steps)]
        table_colors = []
        for step in range(num_token_steps):
            color_row = []
            for beam in range(num_beams):
                token = beam_data[step][beam]
                token_id = dstr.model.decoder.word2id.get(token, 0) # Get ID or default
                color_row.append(cmap(norm(token_id)))
            table_colors.append(color_row)

        # Make the table plot
        table = plt.table(
            cellText=table_data,
            cellColours=table_colors,
            rowLabels=row_headers,
            colLabels=col_headers,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.title(f"Token Chart - Decoder Layer {layer_idx}", pad=-1000)
        plt.tight_layout()

    except Exception as e:
        st.error(f"Error plotting token chart for layer {layer_idx}: {e}")
        st.warning("Intermediate token data structure might be unexpected.")
        print("Problematic token data:", intermediate_tokens_layer_data) # Debug print
        return None

    return fig


# --- Main App Logic ---
st.set_page_config(layout="wide")
st.title("ODEformer Interactive Explorer")

# Load model (cached)
dstr = load_model()

# --- Initialize Session State ---
# This block runs only once or when the app restarts
if 'times' not in st.session_state:
    st.session_state.times = np.linspace(DEFAULT_T_START, DEFAULT_T_END, DEFAULT_SEQ_LENGTH)
    print("Initialized times")
if 'trajectory' not in st.session_state:
    # Calculate initial trajectory based on initial times
    x1_init = np.cos(st.session_state.times)
    y1_init = np.sin(st.session_state.times)
    st.session_state.trajectory = np.stack([x1_init, y1_init], axis=1)
    print("Initialized trajectory")
if 'model_output' not in st.session_state:
    st.session_state.model_output = None # Stores results after running
    print("Initialized model_output")
if 'selected_layer' not in st.session_state:
    st.session_state.selected_layer = 0
    print("Initialized selected_layer")


# --- Sidebar Controls ---
with st.sidebar:
    st.header("Input Parameters")
    # Use values from session state to initialize widgets
    current_seq_length = len(st.session_state.times)
    current_t_start = st.session_state.times.min()
    current_t_end = st.session_state.times.max()

    seq_length = st.slider("Sequence Length", 10, 200, current_seq_length)
    t_start = st.number_input("Start Time", value=current_t_start)
    t_end = st.number_input("End Time", value=current_t_end)

    if st.button("Generate/Update Input Trajectory"):
        print("Updating input trajectory...")
        if t_start >= t_end:
            st.error("Start Time must be less than End Time.")
        else:
            st.session_state.times = np.linspace(t_start, t_end, seq_length)
            # Recalculate trajectory based on new times
            x1 = np.cos(st.session_state.times)
            y1 = np.sin(st.session_state.times)
            st.session_state.trajectory = np.stack([x1, y1], axis=1)
            # MODIFY TRAJECTORY EXAMPLE (uncomment to add perturbation)
            # if len(st.session_state.trajectory) > 29:
            #    st.session_state.trajectory[29] -= 1.0 # Example modification

            st.session_state.model_output = None # Reset model output
            st.session_state.selected_layer = 0 # Reset layer selection
            print("Input trajectory updated. Model output cleared.")
            st.rerun() # Rerun to reflect changes immediately

    st.header("Visualization Options")
    if 'use_scaled_attention' not in st.session_state:
        st.session_state.use_scaled_attention = True

    # Toggle button for attention type
    st.session_state.use_scaled_attention = st.toggle(
        "Use Scaled Attention",
        value=st.session_state.use_scaled_attention,
    )

    st.header("Model Execution")
    if st.button("Run ODEformer Model", key="run_model_button"):
        # Check if trajectory exists
        if 'trajectory' not in st.session_state or st.session_state.trajectory is None:
             st.error("Input trajectory not generated yet. Click 'Generate/Update Input Trajectory' first.")
        else:
            print("Running ODEformer model...")
            with st.spinner("Running ODEformer... This may take a moment."):
                times_to_run = st.session_state.times.copy() # Use current state
                trajectory_to_run = st.session_state.trajectory.copy() # Use current state
                current_seq_length_run = len(times_to_run) # Get actual length

                # --- Run Model ---
                try:
                    # Ensure model is configured to store attentions before fit
                    dstr.store_attentions = True # Redundant if set in load_model, but safe
                    dstr.fit(times_to_run, trajectory_to_run)

                    # --- Predict ---
                    # pred_trajectory = dstr.predict(times_to_run, trajectory_to_run[0])
                    # r2 = r2_score(trajectory_to_run, pred_trajectory) # Calculate R2 score

                    # --- Process & Store Results ---
                    # Ensure attentions and tokens were actually stored
                    if not hasattr(dstr.model.decoder, 'attn_stored_all_tokens') or \
                       not hasattr(dstr.model.decoder, 'intermediate_tokens_all'):
                       raise ValueError("Model did not store attention or intermediate tokens. Check model configuration.")

                    decoder_attentions = dstr.model.decoder.attn_stored_all_tokens
                    scaled_decoder_attentions = dstr.model.decoder.attn_scaled_stored_all_tokens
                    intermediate_tokens = list(dstr.model.decoder.intermediate_tokens_all)

                    # Validate structure before proceeding
                    if not intermediate_tokens:
                         st.warning("Intermediate tokens list is empty.")
                         # Handle this case - maybe skip attention processing or set defaults
                         num_tokens = 0
                         ytick_labels = []
                         decoder_cross_attn = torch.empty(DEC_LAYERS, MODEL_ARGS['beam_size'], TRANS_HEADS, 0, current_seq_length_run) # Empty tensor
                    else:
                        # Calculate num_tokens based on intermediate_tokens list length
                        num_tokens = len(intermediate_tokens)
                        token_ids = range(num_tokens)
                        # Generate ytick labels - Use try-except for safety
                        try:
                             # Assuming the structure: intermediate_tokens[token_idx][beam_idx_or_similar][data_tuple]
                             # The original code used: intermediate_tokens[i][-1][1][-1] -> This seems fragile.
                             # Let's try to get the last token from the first beam if possible.
                             # Adjust this based on the actual structure printed on error.
                             ytick_labels = [f"Tok{i}:{intermediate_tokens[i][1][0]}" for i in range(num_tokens)] # Safer label attempt
                        except Exception as e:
                             st.warning(f"Could not generate ytick labels automatically: {e}. Using default labels.")
                             ytick_labels = [f"Token_{i}" for i in range(num_tokens)]


                        # Process Cross Attention
                        # Size: dec_layers, beam size, heads, num predicted tokens, input sequence length
                        decoder_cross_attn = torch.zeros(DEC_LAYERS, MODEL_ARGS['beam_size'], TRANS_HEADS, num_tokens, current_seq_length_run) # Init with zeros
                        decoder_scaled_cross_attn = torch.zeros(DEC_LAYERS, MODEL_ARGS['beam_size'], TRANS_HEADS, num_tokens, current_seq_length_run) # Init with zeros

                        for token_id in token_ids:
                             if token_id < len(decoder_attentions) and "cross_attention" in decoder_attentions[token_id]:
                                attn_data = decoder_attentions[token_id]["cross_attention"] # [dec_layers, beam, heads, 1, seq_len]
                                scaled_attn_data = scaled_decoder_attentions[token_id]["cross_attention"] # [dec_layers, beam, heads, 1, seq_len]
                                for dec_layer in range(DEC_LAYERS):
                                     # Ensure shapes match before assignment
                                     target_shape = decoder_cross_attn[dec_layer, :, :, token_id:token_id+1, :].shape
                                     source_shape = attn_data[dec_layer].shape
                                     if source_shape == target_shape:
                                         decoder_cross_attn[dec_layer, :, :, token_id:token_id+1, :] = attn_data[dec_layer]
                                         decoder_scaled_cross_attn[dec_layer, :, :, token_id:token_id+1, :] = scaled_attn_data[dec_layer]
                                     else:
                                         # Handle shape mismatch (e.g., if beam size differs) - log warning
                                         print(f"Shape mismatch layer {dec_layer} token {token_id}. Target: {target_shape}, Source: {source_shape}")
                                         # Attempt to assign matching beams/heads if possible, otherwise stays zero
                                         min_beams = min(target_shape[1], source_shape[1])
                                         min_heads = min(target_shape[2], source_shape[2])
                                         decoder_cross_attn[dec_layer, :min_beams, :min_heads, token_id:token_id+1, :] = attn_data[dec_layer, :min_beams, :min_heads, :, :]
                                         decoder_scaled_cross_attn[dec_layer, :min_beams, :min_heads, token_id:token_id+1, :] = scaled_attn_data[dec_layer, :min_beams, :min_heads, :, :]

                             else:
                                 print(f"Attention data missing for token {token_id}") # Debug print


                    # Store everything needed for plotting
                    st.session_state.model_output = {
                        # "pred_trajectory": pred_trajectory,
                        # "r2_score": r2,
                        "decoder_cross_attn": decoder_cross_attn.detach().cpu(), # Store as CPU tensor
                        "decoder_scaled_cross_attn": decoder_scaled_cross_attn.detach().cpu(), # Store as CPU tensor
                        "intermediate_tokens": intermediate_tokens,
                        "ytick_labels": ytick_labels,
                        "times_run": times_to_run, # Store the times used for this run
                        "trajectory_run": trajectory_to_run, # Store the trajectory used for this run
                        "seq_length_run": current_seq_length_run # Store seq length for plotting
                    }
                    st.session_state.selected_layer = 0 # Reset layer selection
                    print("Model run complete. Results stored.")
                    st.success("Model run complete!")
                    st.rerun() # Rerun to display results

                except Exception as e:
                    st.error(f"Error during model execution or processing: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}") # Show detailed error
                    st.session_state.model_output = None # Clear results on error
                    print(f"Error during model run: {e}")


    # Decoder Layer Selector (only show if model has run successfully)
    if st.session_state.model_output:
        st.header("Decoder Layer")
        # Check if intermediate_tokens is not empty before creating slider
        if st.session_state.model_output["intermediate_tokens"]:
             # Layer slider goes from 0 to DEC_LAYERS-1
             st.session_state.selected_layer = st.slider(
                 "Select Decoder Layer to View",
                 min_value=0,
                 max_value=DEC_LAYERS - 1,
                 value=st.session_state.selected_layer, # Use current value
                 key="decoder_layer_slider"
             )
        else:
             st.info("No token data generated, layer selection disabled.")


# --- Main Area Display ---

# Always display the input trajectory plot
st.subheader("1. Input Trajectory")
if 'trajectory' in st.session_state and st.session_state.trajectory is not None:
    fig_input = plot_input_trajectory(st.session_state.times, st.session_state.trajectory)
    st.pyplot(fig_input)
    plt.close(fig_input) # Close figure to free memory
else:
    st.info("Generate an input trajectory using the controls in the sidebar.")


st.subheader("2. Model Results")
if st.session_state.model_output:
    results = st.session_state.model_output
    selected_layer = st.session_state.selected_layer
    seq_length_run = results['seq_length_run'] # Get seq length from stored results

    # st.metric("R2 Score", f"{results['r2_score']:.4f}")

    # # Predicted Trajectory Plot
    # st.write("---")
    # st.subheader("Predicted vs. Input Trajectory")
    # fig_pred = plot_predicted_trajectory(results['times_run'], results['trajectory_run'], results['pred_trajectory'])
    # st.pyplot(fig_pred)
    # plt.close(fig_pred)

    # Attention and Token Plots (conditionally based on selected layer)

    # Check if intermediate_tokens has data for the layer
    if results['intermediate_tokens'] and len(results['intermediate_tokens']) > selected_layer:
        token_data_layer = results['intermediate_tokens'][selected_layer]
        fig_token = plot_single_token_chart(token_data_layer, dstr, selected_layer)
        if fig_token: # Only plot if figure was generated
             st.pyplot(fig_token)
             plt.close(fig_token)
    else:
        st.warning(f"Intermediate token data not available for layer {selected_layer}.")

    st.subheader(f"Analysis for Decoder Attention")

    # num_tokens = len(intermediate_tokens)
    # try:
    #     ytick_labels = [f"Tok{i}:{intermediate_tokens[i][1][0]}" for i in range(num_tokens)]  # Safer label attempt
    # except Exception as e:
    #     st.warning(f"Could not generate ytick labels automatically: {e}. Using default labels.")
    #     ytick_labels = [f"Token_{i}" for i in range(num_tokens)]

    # Replace the current plot_single_cross_attention implementation with:
    if results['decoder_cross_attn'] is not None and results['decoder_cross_attn'].shape[0] > selected_layer:
        # Create a container for the interactive plot
        attn_container = st.container()

        # Create dictionaries with all available layers (not just the selected one)
        all_decoder_layers = {i: results['decoder_cross_attn'][i, BEAM_IDX_TO_PLOT] for i in
                              range(results['decoder_cross_attn'].shape[0])}
        all_scaled_decoder_layers = {i: results['decoder_scaled_cross_attn'][i, BEAM_IDX_TO_PLOT] for i in
                                     range(results['decoder_scaled_cross_attn'].shape[0])}

        # Get list of all available layers
        available_layers = list(range(results['decoder_cross_attn'].shape[0]))

        # print("all_decoder_layers")
        # print(all_decoder_layers)


        # Use the interactive Plotly version that has built-in toggle buttons for both layers and attention types
        with attn_container:
            fig_attn = plot_cross_attention_interactive(
                all_decoder_layers,  # Pass all layers
                all_scaled_decoder_layers,  # Pass all scaled layers
                results['times_run'],
                # ytick_labels,
                seq_length_run,
                available_layers  # Pass all available layers
            )
            st.plotly_chart(fig_attn, use_container_width=True)

    else:
        st.warning(f"Cross-attention data not available for layer {selected_layer}.")

else:
    st.info("Click 'Run ODEformer Model' in the sidebar to generate and view results.")
