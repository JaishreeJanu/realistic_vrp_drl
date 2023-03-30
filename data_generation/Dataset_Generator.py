



class VRPDataset(Dataset):  
        
    def __init__(self, graph_size=10, n_instances=5, address="Berlin", shop_type="supermarket"):
      #super().__init__()
      self.vrpdataset = generate_graphs(address, shop_type, graph_size, n_instances)
    
    def __len__(self):
      return len(self.vrpdataset)

    def __getitem__(self, index):
        data = self.vrpdataset[index]
        return data

    def sample_graphs_viz(self, n_sample=5):
        sampled_graphs = random.sample(self.vrpdataset, n_sample)
        
        # define subplot grid
        fig, axs = plt.subplots(nrows=n_sample, ncols=1, figsize=(6,20))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(f"Sampled {n_sample} graphs from the dataset", fontsize=16, y=0.95)
        
        # loop through graphs and axes
        s_idx = 0
        labels = [f"c_{i}" for i in range(len(sampled_graphs[0]["coordinates"]))]
        labels[0] = "depot"
        for graph, ax in zip(sampled_graphs, axs.ravel()):
            longitudes = [coord[0] for coord in graph["coordinates"]]
            latitudes = [coord[1] for coord in graph["coordinates"]]
            # filter df for ticker and plot on specified axes
            ax.scatter(longitudes, latitudes)

            # chart formatting
            s_idx += 1
            ax.set_title(f"Sample {s_idx}")
            ax.set_xlabel("longitudes")
            ax.set_ylabel("latitudes")

            for i, txt in enumerate(labels):
                ax.annotate(txt, (longitudes[i], latitudes[i]))

        plt.show()
        
        
# dataset = VRPDataset(graph_size=10, n_instances=3)