%function[statsData] = IDcfu_Final(day0, plateN, fileimage)
    %Input: day0, plate# and .tif file
    %This script identifies bacteria colonies in a petri dish. It generates a
    %mask first to avoid the outer part of the plate. Later, segmentation and
    %label identification allows to target even those colonies that are not
    %completely circular. It retrieves a .jpg file with the identifies colonies
    %and a .mat file with a label, morphological and RGB-lab pixel
    %information. 
    %Last version 25.05.2020

    %clear

    %to test without function
    day0 = '200717';
    plateN = '1';
    fileimage = 'i1_d25_10µl-nr1';
    
    %Pixel transformation from DistancePix to cm
    %pixel_size=1/DistancePix;
    %1inch x 96 pixels; 1inch = 2.54cm
    pixel_size=2.54/96; %cm

    %Information required to generate a label
    day = day0; 
    %Replace plateN it with n counter
    plate = plateN;

    %Read files. Batch of pictures from a plate taken along several days
    file= fileimage;
    %16-bit file, RGB range 0-2500
    I = imread(file, 'tif');
    %Turn into 8-bit, RGB range 0-255 with a simple division
    I = uint8(I/257);

    %%
    %%Remove uninterested region from the image 
    %Get the image size to remove the rim of the petri dish
    imageSize = size(I);
    %center and radius of circle ([c_col, c_row, r]). I set my center at
    %[y0,x0] = [1040, 1015] and r = 845
    ci = [1040, 1015, 845]; 
    %Make a grid the same dimensions as the original image
    [xx,yy] = ndgrid((1:imageSize(1))-ci(1),(1:imageSize(2))-ci(2));
    %Make a mask that will turn black all the area outside the plate by
    %fitting a circle with the size of the plate
    mask = uint8((xx.^2 + yy.^2)<ci(3)^2);
    %Generate the new image, cropped imaged after the mask is applied
    croppedImage = uint8(true(size(I)));
    croppedImage(:,:,1) = I(:,:,1).*mask;
    croppedImage(:,:,2) = I(:,:,2).*mask;
    croppedImage(:,:,3) = I(:,:,3).*mask;

%Remove comments if you want to print the crooped image
%     figure
%     imshow(croppedImage)

%%
    %The red channel was the most informative one, therefore for colony identification
    %I decided to only take information from the red channel
    
    %Correct non-uniform ilumination. Adjust specific range if needed
    rgbI = imadjust(croppedImage(:,:,1), [0 0.60],[]);

%Remove comments if you want to print and save the image after  the crooped image
%     figure
%     imshow(rgbI)
%     print(strcat(filename{1},'-adjust'),'-dpng');
%     close;
    
%%
    %There are two types of colonies on the plates, ones with higher RGB values
    %than the background and some other with lower RGB values than the
    %background. I implemented a two-step process to identify all of them.
    
    %Filter bright colonies
    rgbBW = rgbI >=200;%imshow(rgbBW)
    %remove connected objects
    rgbI_nobord = imclearborder(rgbBW,8);%imshow(rgbI_nobord)
    %to fill up holes
    rgbI_noholes = imfill(rgbI_nobord,'holes');%imshow(rgbI_final)       
    %smooth object. Avoids extracting background information
    seD = strel('diamond',1);
    rgbI_final = imerode(rgbI_noholes,seD);
    
    
    %Find colonies using boundary.
    %B, returns an array of pixel locations
    %L, label matrix of objects
    %n, number of objects (labels)
    [B1,L1,n1] = bwboundaries(rgbI_final,'noholes');   

    %Filter dark colonies
    rgbBW = rgbI < 50;%imshow(rgbBW)
    %remove connected objects
    rgbI_nobord = imclearborder(rgbBW,8);%imshow(rgbI_nobord)
    %rgbI_final = rgbI_nobord;
    %smooth object. Avoids extracting background information.
    seD = strel('diamond',1);
    rgbI_final = imerode(rgbI_nobord,seD);

    %Find colonies using boundary
    [B2,L2,n2] = bwboundaries(rgbI_final,'noholes');

    %Match both boundaries, most of the time no dark colonies will be
    %identify, therefor only the information from brigth colonies will be
    %used.
    if isempty(n2)
        %BW image
        B = B1;
        L = L1;
        n_colonies = n1;
    else
        %BW image
        B = [B1; B2];
        L = (L1 | L2);
        n_colonies = (n1+n2);
    end 
 
%Remove comments if one wants to check the boundaries found
%     figure
%     imshow(L)
%     hold on
%     for k = 1:length(B)
%        boundary = B{k};
%        plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
%     end
        
%Remove if you want to print and save the BW image got from the boundaries
%function
%     figure
%     imshow(L)
%     print(strcat(filename{1},'-BW'),'-dpng');
%     close;

%%
    %get morphological stats
    stats=  regionprops(L, 'Centroid', 'Area', 'EquivDiameter', 'Perimeter', 'Circularity', 'Eccentricity', 'ConvexHull');
    Centroid = floor(cat(1,stats.Centroid));
    Eccentricity = cat(1,stats.Eccentricity);
    Area = cat(1,stats.Area);
    Diameter = cat(1,stats.EquivDiameter);
    
    %%
%     %Get pixel information per channel RGB and Lab. Use the BW image generated using
%     %boundaries
%     %First turn RGB image to Lab
%     labIm = rgb2lab(I);
%     
%     for k = 1:3
%         rgbI = I(:,:,k);
%         labI = labIm(:,:,k);
%         %Save according to RGB channel
%         if k == 1
%             %Better to have a different name for the different structures
%             statsPixel_red = regionprops(L,rgbI,{'Centroid','PixelValues','MeanIntensity'});
%             statsPixel_L = regionprops(L,labI,{'Centroid','PixelValues','MeanIntensity'});
%             %save(strcat(filename{1},'-pixel_red.mat'), 'statsPixel_red');
%         elseif k == 2
%             %Better to have a different name for the different structures
%             statsPixel_green = regionprops(L,rgbI,{'Centroid','PixelValues', 'MeanIntensity'});
%             statsPixel_a = regionprops(L,labI,{'Centroid','PixelValues','MeanIntensity'});
%             %save(strcat(filename{1},'-pixel_green.mat'), 'statsPixel_green');
%         else
%             %Better to have a different name for the different structures
%             statsPixel_blue = regionprops(L,rgbI,{'Centroid','PixelValues', 'MeanIntensity'});
%             statsPixel_b = regionprops(L,labI,{'Centroid','PixelValues','MeanIntensity'});
%             %save(strcat(filename{1},'-pixel_blue.mat'), 'statsPixel_blue');
%         end
% 
%     end
    
%     %%
%     %Get edge information as peaks along the perimeter
%     nh = length(stats);
%     pks = zeros(nh, 1);
%     h_pks = zeros(nh, 1);
%     for i = 1:nh
%         c = stats(i).Centroid;
%         h = stats(i).ConvexHull;
%         %Save the distance values
%         n = length(h(:,1));
%         d = zeros(n, 1);
%         for j = 1:n
%             d(j) = pdist([c; h(j,:)],'euclidean');
%         end
%         %plot(d)
%         %Curve smoothing
%         x = 0:1:n-1;
%         xx = 0:0.1:n-1;
%         yy = spline(x, d, xx);
%         %plot(x,y,'-',xx,yy)
%         pks(i) = length(findpeaks(yy));
%         h_pks(i) = mean(d);
%     end

    %%
    %Filter 'bad quality data'
    %Filter by area bigger than n and eccentricity, the closes to 1 the more
    %line segment, the closes to 0 the more circular shape
    filter1 = find(Area > 200 & Area < 70000 & Eccentricity < 0.79); 

    %Find the elements close the the petri dish walls
    filter2 = zeros(length(filter1),1);

    %Calculate indexes to plot a circle and compare with the indexes in mask
    th = 0:pi/5:2*pi;
    for i = 1:length(filter1)
        %imshow(croppedImage);
        %Fit a circle
        xunit = floor(Diameter(filter1(i))/2 * cos(th) + Centroid(filter1(i),1));
        yunit = floor(Diameter(filter1(i))/2 * sin(th) + Centroid(filter1(i),2));
        %Find within the boundaries. Check ci variable
        mean_xy = mean(yunit>200 & yunit<1875 & xunit>200 & xunit<1875);
        if mean_xy == 1
           filter2(i) = filter1(i);
        end
    end

    filter2 = nonzeros(filter2);
    
    %%
%     %Get std on pixel values of the selected colonies
%     std_red = zeros(length(filter2),1);
%     std_green = zeros(length(filter2),1);
%     std_blue = zeros(length(filter2),1);
%     std_L = zeros(length(filter2),1);
%     std_a = zeros(length(filter2),1);
%     std_b = zeros(length(filter2),1);
%     
%     %Go to each of the colonies that past the two filters and get the std
%     %values using the information from PixelValues
%     for m = 1:length(filter2)
%         std_red(m) = std(double(statsPixel_red(filter2(m)).PixelValues));
%         std_green(m) = std(double(statsPixel_green(filter2(m)).PixelValues));
%         std_blue(m) = std(double(statsPixel_blue(filter2(m)).PixelValues));  
%         std_L(m) = std(double(statsPixel_L(filter2(m)).PixelValues));
%         std_a(m) = std(double(statsPixel_a(filter2(m)).PixelValues));
%         std_b(m) = std(double(statsPixel_b(filter2(m)).PixelValues));
%     end
%     RGB_std = [std_red, std_green, std_blue];
%     Lab_std = [std_L, std_a, std_b];

    %%
%     %Get data filtered 
%     diameter = floor(Diameter(filter2));
%     centroid = Centroid(filter2,:);
%     area = cat(1,stats(filter2).Area);
%     perimeter = cat(1,stats(filter2).Perimeter);
%     circularity = cat(1,stats(filter2).Circularity);
%     eccentricity = cat(1,stats(filter2).Eccentricity);
%     R_mean = cat(1,statsPixel_red(filter2).MeanIntensity);
%     G_mean = cat(1,statsPixel_green(filter2).MeanIntensity);
%     B_mean = cat(1,statsPixel_blue(filter2).MeanIntensity);
%     RGB_mean = [R_mean, G_mean, B_mean];
%     L_mean = cat(1,statsPixel_L(filter2).MeanIntensity);
%     a_mean = cat(1,statsPixel_a(filter2).MeanIntensity);
%     b_mean = cat(1,statsPixel_b(filter2).MeanIntensity);
%     Lab_mean = [L_mean, a_mean, b_mean];
%     peaks = pks(filter2);
%     h_peaks = h_pks(filter2);

    
   %%
%     %Extract transversal information colony: mean and std
%     %empty array to sabe data
%     RGBt_mean = zeros(length(diameter), 3);
%     RGBt_std = zeros(length(diameter), 3);
%     Labt_mean = zeros(length(diameter), 3);
%     Labt_std = zeros(length(diameter), 3);
%     
%     %Get x and y coordinates, x fixed
%     for j = 1:length(diameter)
%         x = floor(centroid(j,1)-diameter(j)/2:centroid(j,1)+diameter(j)/2);
%         y = repmat(floor(centroid(j,2)), 1, length(x));
%         c = improfile(I,x,y);
%         cLab = improfile(labIm, x, y);
%         %Read channel
%         RGBt_mean(j,1) = mean(c(:,:,1));
%         RGBt_std(j,1) = std(c(:,:,1));
%         %L space
%         Labt_mean(j,1) = mean(cLab(:,:,1));
%         Labt_std(j,1) = std(cLab(:,:,1));
%         %Green channel
%         RGBt_mean(j,2) = mean(c(:,:,2));
%         RGBt_std(j,2) = std(c(:,:,2));
%         %a space
%         Labt_mean(j,2) = mean(cLab(:,:,2));
%         Labt_std(j,2) = std(cLab(:,:,2));
%         %Blue channel
%         RGBt_mean(j,3) = mean(c(:,:,3));
%         RGBt_std(j,3) = std(c(:,:,3));
%         %b space
%         Labt_mean(j,3) = mean(cLab(:,:,3));
%         Labt_std(j,3) = std(cLab(:,:,3));
%     
%     %Check transversal fixing y
%     %     x = repmat(floor(centroid(j,1)), 1, length(x));
%     %     y = floor(centroid(j,2)-diameter(1)/2:centroid(j,2)+diameter(1));
%     %     cy = improfile(I,x,y);
% 
%         %Print intensity plots
%     %     figure
%     %     improfile(I,x,y); grid on
%     end

%%
    %%Plot the colonies found and save data
    %At this step labels are generated
    colony = strings(length(diameter),1); 
    label = strings(length(diameter),1);
    sample = strings(length(diameter),1);
    %s = split(fileimage, '_');
    s = fileimage;
    ID = strings(length(diameter),1);

    %Give labels
    for i = 1:length(diameter)
        colony(i) = int2str(i);
        label(i)= strcat(day,'-',plate,'-',int2str(i));
        %sample(i) = s{3};
        sample(i) = s;
        ID(i)=strcat(plate,'-',int2str(i));
    end
    
    % 
    figure
    imshow(I);
    viscircles(centroid,diameter/2,'EdgeColor','b');
    text(centroid(:,1), centroid(:,2), colony);
    print(strcat(file,'-IDs'),'-dpng');
    close;
    
    %%
    %Check how to save the data, all the variables bwboundaries, regiongprops
    %morphological and pixel values
    %Save data
    statsData = struct('label', label, 'sample', sample, 'ID', ID, 'centroid', centroid,...
        'area', area*pixel_size, 'diameter', diameter*pixel_size,'perimeter', perimeter*pixel_size,...
        'peaks', peaks, 'height_peaks', h_peaks, 'circularity', circularity,...
        'eccentricity', eccentricity,'RGB_mean', RGB_mean, 'RGB_std', RGB_std,...
        'RGBt_mean', RGBt_mean, 'RGBt_std', RGBt_std, 'Lab_mean', Lab_mean,...
        'Lab_std', Lab_std, 'Labt_mean', Labt_mean, 'Labt_std', Labt_std);
    save(strcat(file,'-data.mat'), 'statsData');
    
    %n_colonies = length(area);
%end