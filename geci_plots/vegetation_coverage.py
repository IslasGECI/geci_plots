def transform_quadrants_vegetation_data(quadrants_vegetation_data):
    return quadrants_vegetation_data.groupby(["Cerco", "Cuadrante"], as_index=False).agg(
        {"Cobertura": "sum", "Altura": "mean"}
    )
