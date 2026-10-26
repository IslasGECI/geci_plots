def transform_quadrants_vegetation_data(quadrants_vegetation_data):
    coverage_columns = [
        "Waltheria indica",
        "Tribulus cistoides",
        "Canavalia rosea",
        "Suelo",
    ]
    return quadrants_vegetation_data.pivot_table(
        index="Cuadrante", columns="Especie", values="Cobertura", fill_value=0
    )[coverage_columns]
