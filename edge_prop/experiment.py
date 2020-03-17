from edge_prop.data_loader import DataLoader


def run(datasets, models):
    for dataset in datasets:
        data_loader = DataLoader(dataset)
        graph_nx, y, mask = data_loader.load_data()

        for model in models:
            model.fit(graph_nx, y, mask)
            y_pred = model.predict(graph_nx)

if __name__ == '__main__':
    run()