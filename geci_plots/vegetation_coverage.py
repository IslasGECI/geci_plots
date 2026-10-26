def transform_quadrants_vegetation_data(quadrants_vegetation_data):
    return quadrants_vegetation_data.pivot_table(
        index="Cuadrante", columns="Especie", values="Cobertura", fill_value=0
    )
