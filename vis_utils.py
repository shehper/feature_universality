# def plot_grid_plotly(scatter_data, corr_coefs, subplot_titles,
#                       name = "all",
#                       grid_length = grid_length, grid_titles = grid_titles):
#     assert name in ["all", "masked", "all_relative", "masked_relative"]
#     fig = sp.make_subplots(rows=grid_length, cols=grid_length, subplot_titles=subplot_titles, 
#                            row_titles=grid_titles, column_titles=grid_titles,
#                        x_title="Maximum Feature Activation", y_title="Maximum Activation Similarity",
#                        horizontal_spacing=0.05, vertical_spacing=0.05)

#     i = 0
#     for row in range(1, grid_length + 1):
#         for col in range(1, grid_length + 1):
#             if row == col:
#                 continue

#             x, y = scatter_data[i]
#             corr_coef = corr_coefs[i]

#             fig.add_trace(
#                 go.Scatter(x=x.numpy(), y=y.numpy(), mode='markers', name=""),
#                 row=row, col=col
#             )
#             i += 1


#     for i in range(grid_length):
#         fig.layout.annotations[grid_length**2 + i].update(y=1.025)

#     fig.update_layout(
#         title_text="Feature Importance (x-axis) vs Universality (y-axis)",
#         showlegend=False,
#         height=2000,
#         width=2000,
#     )

#     os.makedirs("plots", exist_ok=True)
#     fig.write_html(f"plots/imp_vs_uni_{name}.html")
#     print(f"Saved {name} plot to plots/imp_vs_uni_{name}.html")

#     return fig

# ## Use as 
# fig = plot_grid_plotly(all_scatter_data, all_corr_coefs, all_subplot_titles, name = "all_relative")
# fig = plot_grid_plotly(masked_scatter_data, masked_corr_coefs, masked_subplot_titles, name = "masked_relative")