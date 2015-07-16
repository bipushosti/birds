function plot_flightData(flight_array)
  
    figure
    hold on
    sizes = size(flight_array);
    num_rows = sizes(1);
    lw = dlmread('lw_crop.txt');
    lw = lw';
  
    contour(lw);
       
    row_val = flight_array(1:num_rows,1);
    col_val = flight_array(1:num_rows,2);

    for k = 1:num_rows
        view([90 90]);
         scatter(row_val(k),col_val(k));
        pause(0.00000001);
    end
        %pause(0.0000001);
   
end